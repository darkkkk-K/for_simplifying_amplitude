"""Current expression-only actor–critic after a frozen768 encoder."""
from dataclasses import dataclass
import itertools
import math
import torch
from torch import nn
from torch.nn import functional as F
from data.symbolic import UnsupportedExpression, action_features, content_id, pair_is_legal, parse_terms, vocabulary
SCHEMA = 'expression_best_v1'
CONTEXT_FEATURES = ('point4','point5','point6','log_current_terms','log_best_terms')
def _check_counts(n_point, current_terms, best_terms):
    if type(n_point) is not int or n_point not in (4, 5, 6):
        raise ValueError('Explicit n_point4/5/6 required')
    if type(best_terms) is not int or not (best_terms == 0 if current_terms == 0 else 1 <= best_terms <= current_terms):
        raise ValueError('Historical minimum must match a nonterminal/zero expression')


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    token_ids: torch.Tensor
    symbolic: torch.Tensor
    actions: torch.Tensor
    positions: torch.Tensor
    action_features: torch.Tensor
    n_point: int
    best_terms: int
    state_id: str


@dataclass(frozen=True, slots=True)
class EncodedObservation:
    encoded: torch.Tensor
    mask: torch.Tensor
    symbolic: torch.Tensor
    actions: torch.Tensor
    positions: torch.Tensor
    action_features: torch.Tensor
    n_point: int
    best_terms: int
    state_id: str

    @property
    def context(self):
        current = len(self.encoded)
        _check_counts(self.n_point, current, self.best_terms)
        return torch.tensor([float(self.n_point == n) for n in (4, 5, 6)] +
                            [math.log1p(current), math.log1p(self.best_terms)], dtype=torch.float32)

    @property
    def context_id(self):
        return content_id([SCHEMA, self.state_id, self.n_point, self.best_terms])

    def tensor_bytes(self):
        return sum(getattr(self, name).numel()*getattr(self, name).element_size()
                   for name in ('encoded', 'mask', 'symbolic', 'actions', 'positions', 'action_features'))


def build_observation(tokens, n_point, best_terms, *, max_terms=128, max_tokens=128, max_actions=32768):
    """Enumerate structural actions without manufacturing a dummy budget."""
    _check_counts(n_point, len(tokens), best_terms)
    if min(max_terms, max_tokens, max_actions) <= 0:
        raise ValueError('Positive capacity limits required')
    if len(tokens) > max_terms or max(map(len, tokens), default=0) > max_tokens:
        raise UnsupportedExpression('Network capacity exceeded; no truncation')
    terms = parse_terms(tokens)
    ids = torch.zeros(len(tokens), max(1, max(map(len, tokens), default=0)), dtype=torch.long)
    symbolic = torch.zeros(len(tokens), 32)
    logs = [math.log(abs(term.coefficient)) for term in terms]
    mean_log = sum(logs)/len(logs) if logs else 0.
    indices, positions, features = [], [], []
    vocab = vocabulary()
    for t, (raw, term) in enumerate(zip(tokens, terms)):
        ids[t, :len(raw)] = torch.tensor([vocab[x] for x in raw])
        for factor in term.factors:
            symbolic[t, factor.token_id-16] = math.log1p(factor.power)
        symbolic[t, 30] = 1 if term.coefficient > 0 else -1
        symbolic[t, 31] = logs[t]-mean_log
        for i, j in itertools.combinations(range(len(term.factors)), 2):
            a, b = term.factors[i], term.factors[j]
            if not pair_is_legal(a, b, n_point):
                continue
            if len(indices) >= max_actions:
                raise UnsupportedExpression('Action capacity exceeded; no silent filtering')
            indices.append((t, i, j))
            positions.append((a.token_position, b.token_position))
            features.append(action_features(a, b))
    return PolicyObservation(ids, symbolic, torch.tensor(indices, dtype=torch.long).reshape(-1, 3),
                             torch.tensor(positions, dtype=torch.long).reshape(-1, 2),
                             torch.tensor(features, dtype=torch.float32).reshape(-1, 69),
                             n_point, best_terms, content_id(tokens))

class RestrictedActorCritic(nn.Module):

    def __init__(self, encoder, *, enabled=False, top_k=8):
        super().__init__()
        if not enabled:
            raise RuntimeError('Early fusion branch is off')
        if type(top_k) is not int or top_k <= 0 or encoder.emb_dim != 768:
            raise ValueError('Expected768 encoder and positive top_k')
        self.encoder = encoder.requires_grad_(False).eval()
        self.top_k = top_k
        self.token_project = nn.Linear(768, 512, bias=False)
        self.symbolic_encoder = nn.Sequential(nn.Linear(35, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(576, 512), nn.GELU(), nn.LayerNorm(512))
        self.term_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(512, 8, 1024, dropout=0.0, activation='gelu', batch_first=True), 1, enable_nested_tensor=False)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.bracket_project = nn.Linear(512, 256, bias=False)
        self.bracket_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 8, 512, dropout=0.0, activation='gelu', batch_first=True), 1, enable_nested_tensor=False)
        self.term_adapter = nn.Sequential(nn.Linear(576, 64), nn.GELU(), nn.LayerNorm(64))
        self.bracket_adapter = nn.Sequential(nn.Linear(256, 32), nn.GELU())
        self.action_head = nn.Sequential(nn.Linear(305, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        self.value_head = nn.Sequential(nn.Linear(588, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        for name, module in self.named_modules():
            if name == 'encoder' or name.startswith('encoder.'):
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
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.action_head[-1].weight, std=0.01)
        charges = torch.zeros(30, 7)
        for token, idx in vocabulary().items():
            if 16 <= idx <= 45:
                charges[idx - 16, 0] = 1
                for label in token[1:-1]:
                    charges[idx - 16, int(label)] = 1 if token[0] == '<' else -1
        self.register_buffer('charges', charges)
        self.action_head[0] = nn.Linear(298, 64)
        self.value_head[0] = nn.Linear(581, 128)
        for layer in (self.action_head[0], self.value_head[0]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def trainable_parameters(self):
        return (p for name, p in self.named_parameters() if not name.startswith('encoder.'))

    def small_state_dict(self):
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items() if not name.startswith('encoder.')}

    def load_small_state(self, state):
        expected = self.small_state_dict()
        if set(expected) != set(state) or any((state[k].shape != expected[k].shape for k in expected)):
            raise ValueError('Policy state mismatch')
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any((not k.startswith('encoder.') for k in result.missing_keys)):
            raise ValueError('Unexpected missing policy parameter')
    observation_schema = SCHEMA

    @torch.no_grad()
    def prepare(self, observation):
        if type(observation) is not PolicyObservation:
            raise TypeError('Only PolicyObservation is accepted; no environment context')
        if not len(observation.actions):
            return None
        device = self.token_project.weight.device
        self.encoder.eval()
        ids = observation.token_ids.to(device)
        encoded = self.encoder(ids.unsqueeze(0), RL=True)
        if encoded.shape != (*ids.shape, 768) or not bool(torch.isfinite(encoded).all()):
            raise ValueError('Invalid pretrained encoding')
        return EncodedObservation(encoded.cpu().clone(), (ids != 0).cpu(), observation.symbolic.clone(), observation.actions.clone(), observation.positions.clone(), observation.action_features.clone(), observation.n_point, observation.best_terms, observation.state_id)

    def _forward(self, states, *, with_value, return_hidden=False):
        if not states or any((type(s) is not EncodedObservation or not len(s.actions) for s in states)):
            raise TypeError('Expected nonempty restricted encoded observations')
        device = self.token_project.weight.device
        B, T, S = (len(states), max((len(s.encoded) for s in states)), max((s.encoded.shape[1] for s in states)))
        E = torch.zeros(B, T, S, 768, device=device)
        mask = torch.zeros(B, T, S, dtype=torch.bool, device=device)
        phi = torch.zeros(B, T, 32, device=device)
        for b, state in enumerate(states):
            t, s = state.mask.shape
            E[b, :t, :s] = state.encoded.to(device)
            mask[b, :t, :s] = state.mask.to(device)
            phi[b, :t] = state.symbolic.to(device)
        real = mask.any(-1)
        points = F.one_hot(torch.tensor([s.n_point - 4 for s in states], device=device), 3).float()[:, None].expand(-1, T, -1)
        X = self.token_project(E) * mask.unsqueeze(-1)
        P = F.normalize(X.sum(2), dim=-1)
        Z = self.symbolic_encoder(torch.cat([phi, points], -1)) * real.unsqueeze(-1)
        C = self.term_transformer(self.fusion(torch.cat([P, Z], -1)), src_key_padding_mask=~real).masked_fill(~real.unsqueeze(-1), 0.0)
        V = self.bracket_project(X + self.alpha * C.unsqueeze(2)).reshape(B * T, S, 256)
        real_flat = real.flatten()
        H_flat = V.new_zeros(B * T, S, 256)
        H_flat[real_flat] = self.bracket_transformer(V[real_flat], src_key_padding_mask=~mask.reshape(B * T, S)[real_flat])
        H = H_flat.reshape(B, T, S, 256).masked_fill(~mask.unsqueeze(-1), 0.0)
        U = self.term_adapter(torch.cat([C, Z], -1)) * real.unsqueeze(-1)
        denom = real.sum(1, keepdim=True)
        G = U.sum(1) / denom
        context = torch.stack([s.context for s in states]).to(device)
        values = self.value_head(torch.cat([C.sum(1) / denom, Z.sum(1) / denom, context], -1)).squeeze(-1) if with_value else None
        counts = [len(s.actions) for s in states]
        ab = torch.repeat_interleave(torch.arange(B, device=device), torch.tensor(counts, device=device))
        terms = torch.cat([s.actions[:, 0] for s in states]).to(device)
        positions = torch.cat([s.positions for s in states]).to(device)
        left, right = (F.normalize(H[ab, terms, positions[:, i]], dim=-1) for i in (0, 1))
        L, R = (self.bracket_adapter(left), self.bracket_adapter(right))
        pair = torch.cat([L + R, (L - R).abs(), L * R], -1)
        psi = torch.cat([s.action_features for s in states]).to(device)
        action_input = torch.cat([U[ab, terms], G[ab], pair, psi, context[ab]], -1)
        flat = self.action_head(action_input).squeeze(-1)
        logits = nn.utils.rnn.pad_sequence(flat.split(counts), batch_first=True)
        action_mask = torch.arange(logits.shape[1], device=device)[None] < torch.tensor(counts, device=device)[:, None]
        if not bool(torch.isfinite(logits).all()) or (values is not None and (not bool(torch.isfinite(values).all()))):
            raise ValueError('Nonfinite policy/value output')
        result = (logits, action_mask, values)
        return (*result, {'Z': Z, 'C': C, 'H': H}) if return_hidden else result

    def forward(self, states, *, return_hidden=False):
        return self._forward(states, with_value=True, return_hidden=return_hidden)

    def policy_logits(self, states):
        logits, mask, _ = self._forward(states, with_value=False)
        return (logits, mask)

    def checkpoint_payload(self, **metadata):
        return {'observation_schema': SCHEMA, 'context_features': list(CONTEXT_FEATURES), 'encoder_hidden_dim': 768, 'top_k': self.top_k, 'state_dict': self.small_state_dict(), 'metadata': metadata}

    def load_checkpoint_payload(self, payload):
        if payload.get('observation_schema') != SCHEMA or payload.get('context_features') != list(CONTEXT_FEATURES) or payload.get('encoder_hidden_dim') != 768 or (payload.get('top_k') != self.top_k):
            raise ValueError('Checkpoint observation contract mismatch; no implicit legacy conversion')
        self.load_small_state(payload['state_dict'])

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

def select_argmax(logits, mask):
    if logits.ndim != 2 or logits.shape[0] != 1 or mask.shape != logits.shape or mask.dtype != torch.bool:
        raise ValueError('Single-state logits/mask required')
    if not bool(mask.any()) or not bool(torch.isfinite(logits[mask]).all()):
        raise ValueError('Empty or invalid legal support')
    scores = logits.detach().masked_fill(~mask, -torch.inf)
    index = int(scores[0].argmax())
    return index, int((mask[0] & (scores[0] == scores[0, index])).sum())
