"""Fresh trainable policy after a frozen pretrained encoder; explicit opt-in."""
from dataclasses import dataclass, replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from .legal_actions import content_id, vocabulary


@dataclass(frozen=True)
class EncodedState:
    encoded: torch.Tensor
    mask: torch.Tensor
    symbolic: torch.Tensor
    actions: torch.Tensor
    positions: torch.Tensor
    action_features: torch.Tensor
    context: torch.Tensor
    potential: float
    n_point: int
    state_id: str
    context_id: str

    def tensor_bytes(self):
        return sum(x.numel()*x.element_size() for x in vars(self).values() if isinstance(x, torch.Tensor))


def condition(state, progress):
    from .single24_policy import CUTS
    if not 0 <= progress.step < 24:
        raise ValueError("No policy decision at terminal horizon")
    cut = next(c for c in CUTS if c > progress.step)
    env = [float(state.n_point == n) for n in (4, 5, 6)] + [math.log1p(len(state.encoded)), (24-progress.step)/24, math.log1p(24)]
    history = [math.log1p(progress.initial), progress.best/progress.initial, progress.best_stage/6,
               progress.paid_bonus*progress.initial/.5, (cut-progress.step)/24, state.potential]
    return replace(state, context=torch.tensor(env+history, dtype=torch.float32),
                   context_id=content_id([state.state_id, state.n_point, vars(progress)]))


class EarlyFusionActorCritic(nn.Module):
    def __init__(self, encoder, *, enabled=False, top_k=16):
        super().__init__()
        if not enabled:
            raise RuntimeError("Early fusion branch is off")
        if type(top_k) is not int or top_k <= 0 or encoder.emb_dim != 768:
            raise ValueError("Expected768 encoder and positive top_k")
        self.encoder = encoder.requires_grad_(False).eval()
        self.top_k = top_k
        self.token_project = nn.Linear(768, 512, bias=False)
        self.symbolic_encoder = nn.Sequential(nn.Linear(35, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(576, 512), nn.GELU(), nn.LayerNorm(512))
        self.term_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            512, 8, 1024, dropout=0., activation="gelu", batch_first=True), 1, enable_nested_tensor=False)
        self.alpha = nn.Parameter(torch.tensor(1.))
        self.bracket_project = nn.Linear(512, 256, bias=False)
        self.bracket_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            256, 8, 512, dropout=0., activation="gelu", batch_first=True), 1, enable_nested_tensor=False)
        self.term_adapter = nn.Sequential(nn.Linear(576, 64), nn.GELU(), nn.LayerNorm(64))
        self.bracket_adapter = nn.Sequential(nn.Linear(256, 32), nn.GELU())
        self.action_head = nn.Sequential(nn.Linear(305, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        self.value_head = nn.Sequential(nn.Linear(588, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        # Do not recursively initialize or touch any module within the encoder.
        for name, module in self.named_modules():
            if name == "encoder" or name.startswith("encoder."):
                continue
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight); nn.init.zeros_(module.bias)
        nn.init.normal_(self.action_head[-1].weight, std=.01)
        charges = torch.zeros(30, 7)
        for token, idx in vocabulary().items():
            if 16 <= idx <= 45:
                charges[idx-16, 0] = 1
                for label in token[1:-1]:
                    charges[idx-16, int(label)] = 1 if token[0] == '<' else -1
        self.register_buffer("charges", charges)

    def train(self, mode=True):
        super().train(mode); self.encoder.eval(); return self

    def trainable_parameters(self):
        return (p for name, p in self.named_parameters() if not name.startswith("encoder."))

    def small_state_dict(self):
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items()
                if not name.startswith("encoder.")}

    def load_small_state(self, state):
        expected = self.small_state_dict()
        if set(expected) != set(state) or any(state[k].shape != expected[k].shape for k in expected):
            raise ValueError("Policy state mismatch")
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not k.startswith("encoder.") for k in result.missing_keys):
            raise ValueError("Unexpected missing policy parameter")

    @torch.no_grad()
    def prepare(self, batch, progress):
        if batch.token_ids.shape[0] != 1:
            raise ValueError("Encode one state at a time")
        rows = batch.actions
        if not len(rows.action_indices):
            return None
        if not bool(rows.query_allowed.all()):
            raise ValueError("No query-history exclusions in this branch")
        device = self.token_project.weight.device
        ids = batch.token_ids.to(device)
        self.encoder.eval()
        encoded = self.encoder(ids, RL=True)
        mask = ids[0] != 0
        if encoded.shape != (*mask.shape, 768):
            raise ValueError("Encoder shape mismatch")
        phi = batch.term_features[0].to(device)
        length = len(encoded)
        potential = 0.
        if length > 1:
            z = F.normalize((encoded*mask.unsqueeze(-1)).sum(1), dim=-1)
            exponents = phi[:, :30].expm1().round()
            q = exponents @ self.charges
            compatible = (q[:, None] == q[None, :]).all(-1)
            distance = (exponents[:, None]-exponents[None, :]).abs().sum(-1)
            similarity = (z@z.T).clamp(0, 1)*torch.exp(-distance/4)*compatible
            similarity.fill_diagonal_(0.)
            potential = float(similarity.max(-1).values.mean().clamp(0, 1))
        if not bool(torch.isfinite(encoded).all()):
            raise ValueError("Nonfinite encoder")
        state = EncodedState(encoded.cpu().clone(), mask.cpu(), phi.cpu(), rows.action_indices.cpu(),
            rows.token_positions.cpu(), rows.action_features.cpu(), torch.zeros(12), potential,
            int(batch.n_point[0]), batch.state_ids[0], "")
        return condition(state, progress)

    def forward(self, states, *, return_hidden=False):
        if not states or any(len(s.actions) == 0 for s in states):
            raise ValueError("Nonempty states/support required")
        device = self.token_project.weight.device
        B, T, S = len(states), max(len(s.encoded) for s in states), max(s.encoded.shape[1] for s in states)
        E = torch.zeros(B, T, S, 768, device=device)
        mask = torch.zeros(B, T, S, dtype=torch.bool, device=device)
        phi = torch.zeros(B, T, 32, device=device)
        for b, state in enumerate(states):
            t, s = state.mask.shape
            E[b, :t, :s] = state.encoded.to(device)
            mask[b, :t, :s] = state.mask.to(device)
            phi[b, :t] = state.symbolic.to(device)
        real = mask.any(-1)
        npoints = torch.tensor([s.n_point-4 for s in states], device=device)
        points = F.one_hot(npoints, 3).float()[:, None].expand(-1, T, -1)
        X = self.token_project(E)*mask.unsqueeze(-1)
        P = F.normalize(X.sum(2), dim=-1)
        Z = self.symbolic_encoder(torch.cat([phi, points], -1))*real.unsqueeze(-1)
        Fused = self.fusion(torch.cat([P, Z], -1))
        C = self.term_transformer(Fused, src_key_padding_mask=~real).masked_fill(~real.unsqueeze(-1), 0.)
        V = self.bracket_project(X+self.alpha*C.unsqueeze(2)).reshape(B*T, S, 256)
        real_flat = real.flatten()
        H_flat = V.new_zeros(B*T, S, 256)
        H_flat[real_flat] = self.bracket_transformer(V[real_flat], src_key_padding_mask=~mask.reshape(B*T, S)[real_flat])
        H = H_flat.reshape(B, T, S, 256).masked_fill(~mask.unsqueeze(-1), 0.)
        U = self.term_adapter(torch.cat([C, Z], -1))*real.unsqueeze(-1)
        denom = real.sum(1, keepdim=True)
        G = U.sum(1)/denom
        context = torch.stack([s.context for s in states]).to(device)
        values = self.value_head(torch.cat([C.sum(1)/denom, Z.sum(1)/denom, context], -1)).squeeze(-1)
        counts = [len(s.actions) for s in states]
        ab = torch.repeat_interleave(torch.arange(B, device=device), torch.tensor(counts, device=device))
        terms = torch.cat([s.actions[:, 0] for s in states]).to(device)
        positions = torch.cat([s.positions for s in states]).to(device)
        left, right = (F.normalize(H[ab, terms, positions[:, i]], dim=-1) for i in (0, 1))
        L, R = self.bracket_adapter(left), self.bracket_adapter(right)
        pair = torch.cat([L+R, (L-R).abs(), L*R], -1)
        psi = torch.cat([s.action_features for s in states]).to(device)
        action_input = torch.cat([U[ab, terms], G[ab], pair, psi, context[ab]], -1)
        flat_logits = self.action_head(action_input).squeeze(-1)
        logits = nn.utils.rnn.pad_sequence(flat_logits.split(counts), batch_first=True)
        action_mask = torch.arange(logits.shape[1], device=device)[None] < torch.tensor(counts, device=device)[:, None]
        if not bool(torch.isfinite(logits).all() and torch.isfinite(values).all()):
            raise ValueError("Nonfinite actor/critic")
        if return_hidden:
            return logits, action_mask, values, {"Z": Z, "C": C, "H": H}
        return logits, action_mask, values


def topk_log_probs(logits, mask, k):
    """The same dynamic hard-top-k distribution in rollout and PPO updates."""
    if logits.ndim != 2 or mask.shape != logits.shape or mask.dtype != torch.bool or k <= 0:
        raise ValueError("Bad dense logits/mask")
    if not bool(mask.any(-1).all()) or not bool(torch.isfinite(logits[mask]).all()):
        raise ValueError("Empty/nonfinite action support")
    keep = torch.zeros_like(mask)
    for b in range(len(logits)):
        allowed = mask[b].nonzero().flatten()
        order = torch.argsort(logits[b, allowed].detach(), descending=True, stable=True)
        keep[b, allowed[order[:k]]] = True
    values = logits.double().masked_fill(~keep, -torch.inf)
    return F.log_softmax(values, dim=-1), keep


def sample_one(logits, mask, k, generator):
    if logits.shape[0] != 1:
        raise ValueError("Single rollout state required")
    # CPU float64 gives a reproducible categorical draw with explicit generator.
    lp, keep = topk_log_probs(logits.detach().cpu().double(), mask.cpu(), k)
    index = int(torch.multinomial(lp[0].exp(), 1, generator=generator))
    return index, float(lp[0, index]), keep[0].nonzero().flatten().tolist()


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


def update_ppo(model, optimizer, states, records, advantages, targets, *, generator, hit, on_step):
    if len(states) != 384 or len(records) != 384:
        raise ValueError("Expected384 real transitions")
    device = model.token_project.weight.device
    adv = torch.tensor(advantages, dtype=torch.float64, device=device)
    adv = (adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
    targets = torch.tensor(targets, dtype=torch.float64, device=device)
    old = torch.tensor([r['old_logp'] for r in records], dtype=torch.float64, device=device)
    updates = []
    for epoch in (1, 2):
        order = torch.randperm(384, generator=generator).tolist()
        for start in range(0, 384, 16):
            indices = order[start:start+16]
            selected = torch.tensor([records[i]['selected_index'] for i in indices], device=device)
            optimizer.zero_grad(set_to_none=True)
            hit('update_forwards'); hit('encoder_feature_reuses', 16)
            logits, mask, values = model([states[i] for i in indices])
            lp, keep = topk_log_probs(logits, mask, model.top_k)
            chosen = lp[torch.arange(16, device=device), selected]
            if epoch == 1 and start == 0 and not torch.allclose(chosen.detach(), old[indices], atol=1e-5, rtol=0.):
                raise ValueError('Initial PPO minibatch does not reproduce rollout probabilities')
            loss, actor, value, ratio = ppo_loss(chosen, old[indices], adv[indices], values, targets[indices])
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            hit('autograd'); loss.backward()
            gradients = {}
            for name, p in model.named_parameters():
                if not name.startswith('encoder.') and p.grad is not None:
                    group = name.split('.')[0]
                    gradients[group] = gradients.get(group, 0.) + float(p.grad.detach().double().square().sum())
            norm = float(nn.utils.clip_grad_norm_(tuple(model.trainable_parameters()), .5, error_if_nonfinite=True))
            hit('optimizer'); optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in model.trainable_parameters()):
                raise ValueError('Nonfinite parameters')
            supports = [keep[j].nonzero().flatten().tolist() for j in range(16)]
            changed = [supports[j] != records[i]['top_indices'] for j, i in enumerate(indices)]
            retained = torch.isfinite(chosen)
            log_ratio = chosen[retained]-old[indices][retained]
            conditional_kl = float((log_ratio.exp()-1-log_ratio).mean()) if bool(retained.any()) else None
            row = {'epoch': epoch, 'indices': indices, 'batch_size': 16, 'loss': float(loss.detach()),
                'actor_loss': float(actor.detach()), 'value_loss': float(value.detach()), 'gradient_norm': norm,
                'gradient_groups': {k: math.sqrt(v) for k, v in gradients.items()},
                'new_logits': [logits[j, mask[j]].detach().cpu().double().tolist() for j in range(16)],
                'new_top_indices': supports, 'new_chosen_logp': [float(v) if math.isfinite(float(v)) else None for v in chosen.detach().cpu()],
                'values': values.detach().cpu().double().tolist(), 'ratio': ratio.detach().cpu().tolist(),
                'support_changed_fraction': sum(changed)/16,
                'selected_action_excluded_fraction': float((~retained).double().mean()),
                'conditional_approx_kl_retained_samples': conditional_kl,
                'old_to_new_kl_infinite_for_any_state': any(changed),
                'clip_fraction': float(((ratio-1).abs() > .2).double().mean())}
            on_step(row)
            updates.append({k: v for k, v in row.items() if k not in ('new_logits', 'new_top_indices')})
    optimizer.zero_grad(set_to_none=True)
    return updates
