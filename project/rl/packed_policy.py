"""Opt-in legal-action policy using existing frozen layers, not dense action heads."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .legal_actions import ActionTable, PolicyInput, content_id


@dataclass(frozen=True)
class CandidateConfig:
    support: str = "encoder_and_structural"
    scorer: str = "shared_mlp_residual"
    symbolic_features: bool = False
    encoder_threshold: float = 0.
    temperature: float = .15

    def __post_init__(self):
        if self.support not in ("encoder_and_structural", "structural_only"):
            raise ValueError("Unknown candidate support")
        if self.scorer not in ("cosine_reference", "shared_mlp_residual"):
            raise ValueError("Unknown candidate scorer")
        if type(self.symbolic_features) is not bool:
            raise ValueError("symbolic_features must be bool")
        if not math.isfinite(self.encoder_threshold) or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid threshold/temperature")

    @property
    def law_id(self):
        return "packed_legal_joint64_v1:" + content_id(asdict(self))


class FrozenPolicyFeatures(nn.Module):
    """Reuse the named layers of policy_machine2. Never call its forward().

    Constructing this explicit adapter freezes the shared referenced layers.
    An off gate must return the original policy without constructing it.
    """
    def __init__(self, policy):
        super().__init__()
        for name in ("embdding", "projector1", "policy_head", "proj_to_bracket", "bracket_head"):
            setattr(self, name, getattr(policy, name))
        self.register_buffer("term_coupling", policy.term_coupling.detach().clone())
        self.encoder_dim = self.projector1.in_features
        self.term_dim = self.projector1.out_features
        self.bracket_dim = self.proj_to_bracket.out_features
        if self.proj_to_bracket.in_features != self.term_dim:
            raise ValueError("Backbone layer widths disagree")
        self.requires_grad_(False)
        self.eval()
        self.counts = {"backbone_forwards": 0, "encoder_forwards": 0}

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, token_ids):
        self.eval()
        self.counts["backbone_forwards"] += 1
        B, T, S = token_ids.shape
        token_mask = token_ids != 0
        term_mask = token_mask.any(-1)
        real = term_mask.reshape(-1)
        reference = self.projector1.weight
        E = reference.new_zeros(B * T, S, self.encoder_dim)
        if not bool(real.any()):
            return E.reshape(B, T, S, -1), reference.new_zeros(B, T, self.term_dim), reference.new_zeros(B, T, S, self.bracket_dim)
        # No all-padding term is sent to either per-term transformer.
        real_tokens = token_ids.reshape(B * T, S)[real]
        self.counts["encoder_forwards"] += 1
        encoded = self.embdding(real_tokens.unsqueeze(1), RL=True)
        if encoded.shape != (real_tokens.shape[0], S, self.encoder_dim):
            raise ValueError("Unexpected pretrained encoder RL output shape")
        E[real] = encoded
        E = E.reshape(B, T, S, self.encoder_dim)
        X = self.projector1(E) * token_mask.unsqueeze(-1)
        P = F.normalize(X.sum(2), dim=-1)
        C = reference.new_zeros(B, T, self.term_dim)
        nonempty = term_mask.any(-1)
        C[nonempty] = self.policy_head(P[nonempty], src_key_padding_mask=~term_mask[nonempty])
        C = C.masked_fill(~term_mask.unsqueeze(-1), 0.)
        V = self.proj_to_bracket(X + self.term_coupling * C.unsqueeze(2)).reshape(B * T, S, self.bracket_dim)
        H = reference.new_zeros(B * T, S, self.bracket_dim)
        H[real] = self.bracket_head(V[real], src_key_padding_mask=~token_mask.reshape(B * T, S)[real])
        H = H.reshape(B, T, S, self.bracket_dim).masked_fill(~token_mask.unsqueeze(-1), 0.)
        if not all(bool(torch.isfinite(x).all()) for x in (E, C, H)):
            raise ValueError("Nonfinite frozen representations")
        return E, C, H


def segmented_log_softmax(logits, action_ptr):
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise ValueError("Expected finite one-dimensional action logits")
    offsets = action_ptr.detach().cpu().tolist()
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(logits) or any(a > b for a, b in zip(offsets, offsets[1:])):
        raise ValueError("Invalid expression offsets")
    # Both sampling and recomputation use float64 normalization.
    values = logits.double()
    return torch.cat([F.log_softmax(values[start:end], dim=0) for start, end in zip(offsets, offsets[1:])])


@dataclass
class CandidateOutput:
    logits: torch.Tensor
    log_probs: torch.Tensor
    base_scores: torch.Tensor
    actions: ActionTable
    context_ids: tuple[str, ...]
    law_id: str

    def actions_for(self, batch_index=0):
        start, end = self.actions.action_ptr[batch_index:batch_index + 2].detach().cpu().tolist()
        return tuple((t, (i, j)) for t, i, j in self.actions.action_indices[start:end].detach().cpu().tolist())

    def score_map(self, batch_index=0):
        start, end = self.actions.action_ptr[batch_index:batch_index + 2].detach().cpu().tolist()
        return dict(zip(self.actions_for(batch_index), self.log_probs[start:end].exp().detach().cpu().tolist()))


class PackedActionPolicy(nn.Module):
    def __init__(self, backbone, config=None, *, enabled=False):
        super().__init__()
        if not enabled:
            raise RuntimeError("Packed policy is off; use the explicit factory gate")
        self.config = config or CandidateConfig()
        self.backbone = backbone
        self.term_adapter = nn.Sequential(nn.Linear(backbone.term_dim + 32, 64), nn.GELU(), nn.LayerNorm(64))
        self.bracket_adapter = nn.Sequential(nn.Linear(backbone.bracket_dim, 32), nn.GELU())
        self.action_scorer = nn.Sequential(nn.Linear(299, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.action_scorer[-1].weight)
        nn.init.zeros_(self.action_scorer[-1].bias)
        self.counts = {"policy_forwards": 0, "head_forwards": 0, "empty_forwards": 0}

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def adapter_parameters(self):
        for module in (self.term_adapter, self.bracket_adapter, self.action_scorer):
            yield from module.parameters()

    def adapter_state_dict(self):
        return {name: tensor.detach().cpu().clone() for name, tensor in self.state_dict().items()
                if not name.startswith("backbone.")}

    def load_adapter_state_dict(self, state):
        expected = self.adapter_state_dict()
        if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected):
            raise ValueError("Adapter checkpoint keys/shapes do not match")
        if any(not bool(torch.isfinite(v).all()) for v in state.values()):
            raise ValueError("Nonfinite adapter checkpoint")
        for prefix, module in (("term_adapter.", self.term_adapter), ("bracket_adapter.", self.bracket_adapter), ("action_scorer.", self.action_scorer)):
            module.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)

    def forward(self, batch: PolicyInput):
        self.counts["policy_forwards"] += 1
        if batch.token_ids.device != self.term_adapter[0].weight.device:
            raise ValueError("Move PolicyInput to the policy device first")
        B = batch.token_ids.shape[0]
        rows = batch.actions
        keep = rows.query_allowed.clone()
        if bool(keep.any()):
            E, C, H = self.backbone(batch.token_ids)
            if self.config.support == "encoder_and_structural":
                b, t, p = rows.action_batch, rows.action_indices[:, 0], rows.token_positions
                e1, e2 = F.normalize(E[b, t, p[:, 0]], dim=-1), F.normalize(E[b, t, p[:, 1]], dim=-1)
                keep = keep & ((e1 * e2).sum(-1) <= self.config.encoder_threshold)
        rows = rows.select(keep, B)
        if len(rows.action_indices) == 0:
            self.counts["empty_forwards"] += 1
            empty = self.term_adapter[0].weight.new_empty(0)
            return CandidateOutput(empty, empty.double(), empty, rows, batch.context_ids, self.config.law_id)
        b, t, p = rows.action_batch, rows.action_indices[:, 0], rows.token_positions
        left, right = F.normalize(H[b, t, p[:, 0]], dim=-1), F.normalize(H[b, t, p[:, 1]], dim=-1)
        base = (left * right).sum(-1) / self.config.temperature
        logits = base
        if self.config.scorer == "shared_mlp_residual":
            self.counts["head_forwards"] += 1
            term_mask = (batch.token_ids != 0).any(-1)
            phi, psi = batch.term_features, rows.action_features
            if not self.config.symbolic_features:
                phi, psi = torch.zeros_like(phi), torch.zeros_like(psi)
            dtype = C.dtype
            U = self.term_adapter(torch.cat([C, phi.to(dtype)], dim=-1)) * term_mask.unsqueeze(-1)
            G = U.sum(1) / term_mask.sum(1, keepdim=True).clamp_min(1)
            L, R = self.bracket_adapter(left), self.bracket_adapter(right)
            pair = torch.cat([L + R, (L - R).abs(), L * R], dim=-1)
            eta = torch.cat([F.one_hot(batch.n_point - 4, 3).to(dtype),
                             term_mask.sum(-1, keepdim=True).to(dtype).log1p(),
                             (batch.remaining_budget.to(dtype) / batch.initial_budget).unsqueeze(-1),
                             batch.initial_budget.to(dtype).log1p().unsqueeze(-1)], dim=-1)
            combined = torch.cat([U[b, t], G[b], pair, psi.to(dtype), eta[b]], dim=-1)
            if combined.shape[1] != 299:
                raise ValueError("Action feature width changed")
            logits = base + self.action_scorer(combined).squeeze(-1)
        return CandidateOutput(logits, segmented_log_softmax(logits, rows.action_ptr), base,
                               rows, batch.context_ids, self.config.law_id)


def attach_candidate_policy(existing_policy, *, mode="off", config=None):
    if mode == "off":
        return existing_policy
    if mode != "packed":
        raise ValueError("mode must be off or packed")
    device = existing_policy.projector1.weight.device
    return PackedActionPolicy(FrozenPolicyFeatures(existing_policy), config, enabled=True).to(device)
