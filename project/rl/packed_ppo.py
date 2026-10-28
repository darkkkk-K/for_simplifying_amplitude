"""Opt-in PPO-Clip + GAE for ordered legal-action sets; no model/engine I/O.

One sample is one controller decision (up to four ordered WOR actions).
Only frozen representations are cached. Trainable features are recomputed on
every mini-batch. The historical REINFORCE implementation is not modified.
"""
from dataclasses import dataclass, fields, replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from .legal_actions import ActionTable
from .packed_policy import CandidateOutput, segmented_log_softmax
from .packed_sampling import recompute_log_prob


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    minibatch_size: int = 64
    update_epochs: int = 4
    normalize_advantage: bool = True

    def __post_init__(self):
        if not (0 <= self.gamma <= 1 and 0 <= self.gae_lambda <= 1
                and 0 < self.clip_epsilon < 1 and self.value_coef > 0
                and self.learning_rate > 0 and self.max_grad_norm > 0):
            raise ValueError("Invalid PPO coefficients")
        if any(not math.isfinite(x) for x in (self.gamma, self.gae_lambda, self.clip_epsilon,
                                              self.value_coef, self.learning_rate, self.max_grad_norm)):
            raise ValueError("Nonfinite PPO coefficient")
        if any(type(x) is not int or x < 1 for x in (self.minibatch_size, self.update_epochs)):
            raise ValueError("Positive integer batch size and epochs required")


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


@dataclass(frozen=True)
class FrozenDecisionFeatures:
    context: torch.Tensor                 # [T, original term_dim], no padding
    symbolic: torch.Tensor                # [T, 32]
    left: torch.Tensor                    # [M, original bracket_dim], normalized
    right: torch.Tensor
    base: torch.Tensor                    # [M]
    environment: torch.Tensor             # [6]
    value_extra: torch.Tensor             # [5], initial length and history summaries
    actions: ActionTable                  # one expression; same recorded order
    context_id: str
    law_id: str

    def to(self, device):
        return replace(self, **{f.name: getattr(self, f.name).to(device) for f in fields(self)
                               if isinstance(getattr(self, f.name), (torch.Tensor, ActionTable))})

    def tensor_bytes(self):
        tensors = [getattr(self, f.name) for f in fields(self) if isinstance(getattr(self, f.name), torch.Tensor)]
        tensors += [getattr(self.actions, f.name) for f in fields(self.actions)]
        return sum(x.numel() * x.element_size() for x in tensors)


class PackedActorCritic(nn.Module):
    def __init__(self, actor, *, enabled=False):
        super().__init__()
        if not enabled:
            raise RuntimeError("PPO is off; explicitly enable the new branch")
        if actor.config.scorer != "shared_mlp_residual":
            raise ValueError("PPO needs a trainable actor")
        if any(p.requires_grad for p in actor.backbone.parameters()):
            raise ValueError("Pretrained backbone must remain frozen")
        self.actor = actor
        # Separate critic: no value-loss gradients enter actor adapters.
        width = actor.backbone.term_dim + 32 + 6 + 5
        self.critic = nn.Sequential(nn.Linear(width, 128), nn.GELU(),
                                    nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)

    def trainable_parameters(self):
        yield from self.actor.adapter_parameters()
        yield from self.critic.parameters()

    def small_state_dict(self):
        return {"actor": self.actor.adapter_state_dict(),
                "critic": {k: v.detach().cpu().clone() for k, v in self.critic.state_dict().items()}}

    @torch.no_grad()
    def prepare(self, batch, initial_terms):
        """One actual decision, not a terminal state; one frozen forward.

        Re-encoding after collection is counted separately by the runner. Old
        logp/value are checked and frozen BEFORE any optimizer step.
        """
        if batch.token_ids.shape[0] != 1 or type(initial_terms) is not int or initial_terms <= 0:
            raise ValueError("One expression and positive initial term count required")
        E, C, H = self.actor.backbone(batch.token_ids)
        rows = batch.actions
        keep = rows.query_allowed.clone()
        if self.actor.config.support == "encoder_and_structural":
            b, t, p = rows.action_batch, rows.action_indices[:, 0], rows.token_positions
            keep &= (F.normalize(E[b, t, p[:, 0]], dim=-1) *
                     F.normalize(E[b, t, p[:, 1]], dim=-1)).sum(-1) <= self.actor.config.encoder_threshold
        selected = rows.select(keep, 1)
        if not len(selected.action_indices):
            raise ValueError("Terminal states must not be inserted as policy decisions")
        t, p = selected.action_indices[:, 0], selected.token_positions
        left, right = (F.normalize(H[0, t, p[:, k]], dim=-1) for k in (0, 1))
        mask = batch.token_ids[0].ne(0).any(-1)
        length = int(mask.sum())
        if length > initial_terms or not bool(mask[:length].all()) or bool(mask[length:].any()):
            raise ValueError("Expected nonexpanding expression with trailing padding only")
        dtype, device = C.dtype, C.device
        env = torch.cat([F.one_hot(batch.n_point - 4, 3).to(dtype),
                         C.new_tensor([[math.log1p(length)]]),
                         (batch.remaining_budget.to(dtype) / batch.initial_budget).unsqueeze(-1),
                         batch.initial_budget.to(dtype).log1p().unsqueeze(-1)], -1)[0]
        total = len(rows.action_indices)
        extra = torch.tensor([math.log1p(initial_terms), length / initial_terms, math.log1p(total),
                              float((~rows.query_allowed).sum()) / max(1, total),
                              math.log1p(len(selected.action_indices))], dtype=dtype, device=device)
        return FrozenDecisionFeatures(C[0, :length].clone(), batch.term_features[0, :length].to(dtype).clone(),
            left, right, (left * right).sum(-1) / self.actor.config.temperature, env, extra,
            selected, batch.context_ids[0], self.actor.config.law_id).to("cpu")

    def forward(self, states):
        """A genuine mini-batch: vectorized trainable layers, ragged actions.

        Frozen token features are not padded/re-encoded in PPO epochs. No
        trainable U/G/logits are cached. Every sample retains its own softmax.
        """
        if not states or any(s.law_id != self.actor.config.law_id for s in states):
            raise ValueError("Empty batch or different policy law")
        device = next(self.critic.parameters()).device
        states = [s.to(device) for s in states]
        sizes = [len(s.context) for s in states]
        action_sizes = [len(s.base) for s in states]
        if min(sizes + action_sizes) < 1:
            raise ValueError("Only real nonempty decisions belong in PPO batches")
        C = torch.cat([s.context for s in states])
        phi = torch.cat([s.symbolic for s in states])
        psi = torch.cat([s.actions.action_features for s in states]).to(C.dtype)
        if not self.actor.config.symbolic_features:
            phi, psi = torch.zeros_like(phi), torch.zeros_like(psi)
        U = self.actor.term_adapter(torch.cat([C, phi], -1))
        # Small per-expression reductions preserve original summation order.
        G = torch.stack([u.mean(0) for u in U.split(sizes)])
        pooled = torch.stack([torch.cat([c.mean(0), p.mean(0)])
                              for c, p in zip(C.split(sizes), phi.split(sizes))])
        env = torch.stack([s.environment for s in states])
        values = self.critic(torch.cat([pooled, env, torch.stack([s.value_extra for s in states])], -1)).squeeze(-1)
        tables, offset, ptr, term_rows = [], 0, [0], []
        for b, (s, size, count) in enumerate(zip(states, sizes, action_sizes)):
            tables.append(replace(s.actions, action_batch=torch.full_like(s.actions.action_batch, b)))
            term_rows.append(s.actions.action_indices[:, 0] + offset)
            ptr.append(ptr[-1] + count)
            offset += size
        table = ActionTable(**{f.name: torch.tensor(ptr, device=device) if f.name == "action_ptr"
                               else torch.cat([getattr(a, f.name) for a in tables]) for f in fields(ActionTable)})
        left, right = torch.cat([s.left for s in states]), torch.cat([s.right for s in states])
        L, R = self.actor.bracket_adapter(left), self.actor.bracket_adapter(right)
        pair = torch.cat([L + R, (L - R).abs(), L * R], -1)
        b = table.action_batch
        combined = torch.cat([U[torch.cat(term_rows)], G[b], pair, psi, env[b]], -1)
        base = torch.cat([s.base for s in states])
        logits = base + self.actor.action_scorer(combined).squeeze(-1)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Nonfinite critic output")
        output = CandidateOutput(logits, segmented_log_softmax(logits, table.action_ptr), base,
                                 table, tuple(s.context_id for s in states), self.actor.config.law_id)
        return output, values


def clipped_objective(new_logp, old_logp, advantages, values, targets, config):
    """PPO clipped surrogate + critic MSE. Old quantities always detached.

    Entropy coefficient is explicitly zero in this first implementation; the
    entropy of a single categorical is NOT the entropy of an ordered set.
    """
    old_logp, advantages, targets = (x.detach() for x in (old_logp, advantages, targets))
    if new_logp.ndim != 1 or any(x.shape != new_logp.shape for x in (old_logp, advantages, values, targets)):
        raise ValueError("PPO expects matching one-dimensional mini-batches")
    if not all(bool(torch.isfinite(x).all()) for x in (new_logp, old_logp, advantages, values, targets)):
        raise ValueError("Nonfinite PPO input")
    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()
    if not bool(torch.isfinite(ratio).all()):
        raise ValueError("Nonfinite PPO ratio; do not silently clamp logp")
    raw = ratio * advantages
    clipped = ratio.clamp(1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantages
    actor_loss = -torch.minimum(raw, clipped).mean()
    value_loss = (values - targets).square().mean()
    loss = actor_loss + config.value_coef * value_loss
    with torch.no_grad():
        stats = {"actor_loss": float(actor_loss), "value_loss": float(value_loss),
                 "approx_kl": float((ratio - 1 - log_ratio).mean()),
                 "clip_fraction": float(((ratio - 1).abs() > config.clip_epsilon).double().mean()),
                 "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max())}
    return loss, stats


@dataclass
class PPOBuffer:
    states: list
    records: list
    old_logp: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    targets: torch.Tensor
    metadata: list
    consumed: bool = False


@torch.no_grad()
def build_buffer(model, episodes, config, *, hit=lambda *_: None, max_bytes=512 * 1024**2):
    states, records, values, advantages, targets, metadata = [], [], [], [], [], []
    used = 0
    allowed_ends = {"budget", "zero_expression", "no_remaining_structural_actions", "no_remaining_policy_actions"}
    for episode in episodes:
        if episode["reason"] not in allowed_ends:
            raise ValueError("Unsupported/capacity failure is not a successful PPO terminal")
        memory, rounds = episode["memory"], episode["rounds"]
        if len(memory) != len(rounds):
            raise ValueError("Missing decision memory")
        local_values = []
        for t, item in enumerate(memory):
            hit("feature_preparations")
            state = model.prepare(item["batch"].to(next(model.parameters()).device), episode["initial_terms"])
            used += state.tensor_bytes()
            if used > max_bytes:
                raise RuntimeError("Frozen feature buffer memory cap exceeded")
            hit("old_value_forwards")
            out, v = model([state])
            lp = recompute_log_prob(out, 0, item["record"])
            old = lp.new_tensor(item["record"].conditional_log_probs)
            if not torch.allclose(lp, old, atol=1e-5, rtol=0):
                raise ValueError("Policy/support changed before PPO collection finished")
            states.append(state); records.append(item["record"]); local_values.append(float(v[0]))
            metadata.append({"source_id": episode["source_id"], "seed": episode["seed"], "step": t,
                             "context_id": state.context_id, "reward": rounds[t]["reward"],
                             "terminal": t == len(rounds) - 1})
        terminal = [False] * len(rounds)
        if terminal:
            terminal[-1] = True
        a, ret = compute_gae([r["reward"] for r in rounds], local_values, terminal,
                             gamma=config.gamma, gae_lambda=config.gae_lambda)
        values.extend(local_values); advantages.extend(a.tolist()); targets.extend(ret.tolist())
    if not states:
        raise ValueError("Empty rollout buffer; no invented replacement trajectories")
    vec = lambda x: torch.tensor(x, dtype=torch.float64)
    return PPOBuffer(states, records, vec([sum(r.conditional_log_probs) for r in records]),
                     vec(values), vec(advantages), vec(targets), metadata)


def update_ppo(model, optimizer, buffer, config, *, generator, hit=lambda *_: None, on_step=lambda *_: None):
    if buffer.consumed:
        raise ValueError("A completed PPO buffer cannot be reused in another update")
    # Failure consumes it too: do not silently restart a partly updated buffer.
    buffer.consumed = True
    N = len(buffer.states)
    adv = buffer.advantages.detach().clone()
    if config.normalize_advantage:
        adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
    device = next(model.parameters()).device
    parameters = tuple(model.trainable_parameters())
    results = []
    for epoch in range(config.update_epochs):
        order = torch.randperm(N, generator=generator).tolist()
        for start in range(0, N, config.minibatch_size):
            indices = order[start:start + config.minibatch_size]
            optimizer.zero_grad(set_to_none=True)
            hit("update_forwards"); hit("frozen_feature_reuses", len(indices))
            output, value = model([buffer.states[i] for i in indices])
            new_logp = torch.stack([recompute_log_prob(output, j, buffer.records[i]).sum()
                                   for j, i in enumerate(indices)])
            take = lambda tensor: tensor[indices].to(device)
            loss, stats = clipped_objective(new_logp, take(buffer.old_logp), take(adv), value,
                                            take(buffer.targets), config)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite PPO loss")
            hit("autograd"); loss.backward()
            norm = float(nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True))
            hit("optimizer"); optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in parameters):
                raise ValueError("Nonfinite updated parameters")
            row = {"epoch": epoch + 1, "batch_size": len(indices), "indices": indices,
                   "gradient_norm": norm, "loss": float(loss.detach()), **stats}
            results.append(row); on_step(row)
    optimizer.zero_grad(set_to_none=True)
    return results
