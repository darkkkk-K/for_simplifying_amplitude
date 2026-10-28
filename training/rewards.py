"""Training-only checkpoints, potential shaping, and growth punishment."""
from dataclasses import dataclass, replace
from fractions import Fraction
import math
import torch
from torch.nn import functional as F
HORIZON=24
BETA=.4
def checkpoints(horizon=HORIZON, beta=BETA):
    fraction = Fraction(str(beta))
    if type(horizon) is not int or horizon <= 0 or not 0 < fraction < 1:
        raise ValueError('Positive horizon and0<beta<1 required')
    t = 0; cuts = []
    while t < horizon:
        t += ((horizon-t)*fraction.numerator+fraction.denominator-1)//fraction.denominator
        cuts.append(t)
    return tuple(cuts)
@dataclass(frozen=True)
class LongProgress:
    initial: int
    best: int
    step: int = 0
    best_stage: int = 0
    paid_bonus: float = 0.
    horizon: int = 20


@dataclass(frozen=True)
class LongContext:
    progress: LongProgress
    previous_terms: int
    previous_step: int = 0
    last_terms: int = 0
    closed: bool = False
    coefficient: float = 1.5
    terminal_coefficient: float = 0.

    @classmethod
    def start(cls, initial, *, horizon=20, terminal_coefficient=0., enabled=False):
        if not enabled:
            raise RuntimeError('Long60 environment is off')
        if type(initial) is not int or initial <= 0 or horizon not in (20, 32) or terminal_coefficient not in (0., 1.):
            raise ValueError('Invalid experiment reward/horizon configuration')
        return cls(LongProgress(initial, initial, horizon=horizon), initial,
                   last_terms=initial, terminal_coefficient=terminal_coefficient)

    def advance(self, length, phi_before, phi_after, terminal, *, capacity=False):
        p = self.progress; t = p.step+1; cuts = checkpoints(p.horizon)
        if self.closed or not 1 <= t <= p.horizon or type(length) is not int or length < 0:
            raise ValueError('Invalid real transition')
        if (t == p.horizon or length == 0) and not terminal:
            raise ValueError('Missing true termination')
        if not all(math.isfinite(v) and 0 <= v <= 1+1e-6 for v in (phi_before, phi_after)):
            raise ValueError('Invalid frozen potential')
        effective = p.best if capacity else length
        best = min(p.best, effective)
        stage = next(j for j, cut in enumerate(cuts, 1) if t <= cut)
        first = stage if best < p.best else p.best_stage
        progress = (p.initial-best)/p.initial
        bonus = .5/p.initial*progress*((len(cuts)-first)/(len(cuts)-1) if first else 0.)
        settle = terminal or t in cuts
        paid = bonus if settle else p.paid_bonus
        main = (p.best-best)/p.initial
        segment = paid-p.paid_bonus
        potential = .25/p.initial*((0. if terminal else phi_after)-phi_before)
        growth = max(length-self.previous_terms, 0) if settle else 0
        weight = t/p.horizon if settle else 0.
        punishment = weight*(growth/p.initial)
        weighted = -self.coefficient*punishment
        correction = (best-length)/p.initial if terminal else 0.
        weighted_terminal = self.terminal_coefficient*correction
        reward = {'main': main, 'segment': segment, 'potential': potential,
                  'punishment': punishment, 'weighted_punishment': weighted,
                  'terminal_correction': correction, 'weighted_terminal': weighted_terminal,
                  'total': main+segment+potential+weighted+weighted_terminal}
        details = {'settled': settle, 'episode_step': t, 'horizon': p.horizon,
                   'initial_terms': p.initial, 'previous_checkpoint_step': self.previous_step,
                   'previous_checkpoint_terms': self.previous_terms, 'actual_terms': length,
                   'growth': growth, 'time_weight': weight, 'punishment': punishment,
                   'c_punishment': self.coefficient, 'weighted_punishment': weighted}
        after = replace(self, progress=LongProgress(p.initial, best, t, first, paid, p.horizon),
                        previous_terms=length if settle else self.previous_terms,
                        previous_step=t if settle else self.previous_step, last_terms=length, closed=terminal)
        return after, reward, details
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
