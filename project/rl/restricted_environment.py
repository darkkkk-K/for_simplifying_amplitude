"""Training-only reward state. Never imported by restricted inference/policy."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .lambda20_policy import Progress20
from .punishment20 import CheckpointPunishmentState, validate_coefficient
from .single24_policy import RewardConfig


@dataclass(frozen=True, slots=True)
class TrainingContext:
    progress: Progress20
    punishment: CheckpointPunishmentState
    rewards: RewardConfig
    coefficient: float

    @classmethod
    def start(cls, initial_terms, *, coefficient=0., rewards=None, enabled=False):
        if not enabled:
            raise RuntimeError('Separated training state is off')
        return cls(Progress20(initial_terms, initial_terms), CheckpointPunishmentState.start(initial_terms),
                   rewards or RewardConfig(segment_enabled=True, potential_enabled=True), validate_coefficient(coefficient))

    def advance(self, actual_terms, phi_before, phi_after, terminal, *, capacity=False):
        effective = self.progress.best if capacity else actual_terms
        progress, reward = self.progress.advance(effective, phi_before, phi_after, terminal, self.rewards)
        punishment, details = self.punishment.advance(progress.step, actual_terms, terminal, c_punishment=self.coefficient)
        if self.coefficient:
            reward = dict(reward, punishment=details['punishment'], weighted_punishment=details['weighted_punishment'],
                          total=reward['total']+details['weighted_punishment'])
        return TrainingContext(progress, punishment, self.rewards, self.coefficient), reward, details


@torch.no_grad()
def frozen_potential(state, charges, *, device):
    """Same frozen-encoder potential as before, outside the observation object."""
    if state is None or len(state.encoded) <= 1:
        return 0.
    encoded, mask, phi = state.encoded.to(device), state.mask.to(device), state.symbolic.to(device)
    z = F.normalize((encoded*mask.unsqueeze(-1)).sum(1), dim=-1)
    exponents = phi[:, :30].expm1().round()
    q = exponents @ charges.to(device)
    compatible = (q[:, None] == q[None, :]).all(-1)
    distance = (exponents[:, None]-exponents[None, :]).abs().sum(-1)
    similarity = (z@z.T).clamp(0, 1)*torch.exp(-distance/4)*compatible
    similarity.fill_diagonal_(0.)
    return float(similarity.max(-1).values.mean().clamp(0, 1))
