"""Single-action finite-horizon PPO: frozen encoder, explicit trajectory context.

Independent opt-in branch. No checkpoint loads, engine calls or training on import.
"""
from dataclasses import asdict, dataclass, fields, replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from .legal_actions import ActionTable, content_id, vocabulary
from .packed_policy import CandidateOutput, segmented_log_softmax
from .packed_ppo import FrozenDecisionFeatures

CUTS = (10, 16, 20, 22, 23, 24)


@dataclass(frozen=True)
class RewardConfig:
    segment_enabled: bool = False
    potential_enabled: bool = False
    segment_scale: float = .5
    potential_scale: float = .25


@dataclass(frozen=True)
class Progress:
    initial: int
    best: int
    step: int = 0
    best_stage: int = 0
    paid_bonus: float = 0.

    def advance(self, length, phi_before, phi_after, terminal, config):
        if self.initial <= 0 or not 0 <= length or not 0 <= self.step < 24:
            raise ValueError("Invalid finite-horizon progress")
        if not all(math.isfinite(x) and 0 <= x <= 1+1e-6 for x in (phi_before, phi_after)):
            raise ValueError("Potential must be bounded and finite")
        t = self.step + 1
        stage = next(j for j, end in enumerate(CUTS, 1) if t <= end)
        best = min(self.best, length)
        first = stage if best < self.best else self.best_stage
        p = (self.initial-best)/self.initial
        bonus = config.segment_scale/self.initial * p * ((6-first)/5 if first else 0.)
        settle = terminal or t in CUTS
        paid = bonus if settle else self.paid_bonus
        main = (self.best-best)/self.initial
        segment = paid-self.paid_bonus if config.segment_enabled else 0.
        potential = config.potential_scale/self.initial * ((0. if terminal else phi_after)-phi_before) if config.potential_enabled else 0.
        result = Progress(self.initial,best,t,first,paid)
        return result, {"main":main,"segment":segment,"potential":potential,"total":main+segment+potential}


@dataclass(frozen=True)
class SingleFeatures(FrozenDecisionFeatures):
    trajectory: torch.Tensor
    potential: float
    base_context_id: str


def condition(state, progress):
    if progress.step >= 24:
        raise ValueError("Terminal24 state has no policy decision")
    next_cut = next(x for x in CUTS if x > progress.step)
    x = state.context.new_tensor([math.log1p(progress.initial), progress.best/progress.initial,
        progress.best_stage/6, progress.paid_bonus*progress.initial/.5,
        (next_cut-progress.step)/24, state.potential])
    return replace(state,trajectory=x,context_id=content_id([state.base_context_id,asdict(progress)]))


class SingleActorCritic(nn.Module):
    def __init__(self, actor, *, enabled=False):
        super().__init__()
        if not enabled: raise RuntimeError("single24 branch is off")
        self.actor = actor
        self.actor.action_scorer = nn.Sequential(nn.Linear(305,64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.actor.action_scorer[-1].weight); nn.init.zeros_(self.actor.action_scorer[-1].bias)
        self.critic = nn.Sequential(nn.Linear(actor.backbone.term_dim+32+6+5+6,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.critic[-1].weight); nn.init.zeros_(self.critic[-1].bias)
        charges = torch.zeros(30,7)
        for token, idx in vocabulary().items():
            if 16 <= idx <= 45:
                charges[idx-16,0] = 1
                for label in token[1:-1]: charges[idx-16,int(label)] = 1 if token[0] == '<' else -1
        self.register_buffer("charges",charges)
        self.law_id = "single_categorical24_context6_v1:"+actor.config.law_id
        if any(p.requires_grad for p in actor.backbone.parameters()): raise ValueError("Backbone must be frozen")

    def train(self, mode=True):
        super().train(mode); self.actor.backbone.eval(); return self

    def trainable_parameters(self):
        yield from self.actor.adapter_parameters()
        yield from self.critic.parameters()

    def small_state_dict(self):
        return {"actor":self.actor.adapter_state_dict(),"critic":{k:v.detach().cpu().clone() for k,v in self.critic.state_dict().items()}}

    def load_small_state(self, state):
        self.actor.load_adapter_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"],strict=True)

    @torch.no_grad()
    def prepare(self, batch, initial_terms):
        if batch.token_ids.shape[0]!=1 or initial_terms<=0: raise ValueError("Expected one state")
        rows = batch.actions
        if not bool(rows.query_allowed.any()): return None
        E,C,H = self.actor.backbone(batch.token_ids)
        keep = rows.query_allowed.clone()
        if self.actor.config.support == "encoder_and_structural":
            b,t,p = rows.action_batch,rows.action_indices[:,0],rows.token_positions
            keep &= (F.normalize(E[b,t,p[:,0]],dim=-1)*F.normalize(E[b,t,p[:,1]],dim=-1)).sum(-1) <= self.actor.config.encoder_threshold
        selected = rows.select(keep,1)
        if not len(selected.action_indices): return None
        mask = batch.token_ids[0].ne(0); real = mask.any(-1); length = int(real.sum())
        if not bool(real[:length].all()) or bool(real[length:].any()): raise ValueError("Padding is not trailing")
        t,p = selected.action_indices[:,0],selected.token_positions
        left,right = (F.normalize(H[0,t,p[:,i]],dim=-1) for i in (0,1))
        phi = batch.term_features[0,:length].to(C.dtype)
        potential = 0.
        if length > 1:
            z = F.normalize((E[0,:length]*mask[:length].unsqueeze(-1)).sum(1),dim=-1)
            exponents = phi[:,:30].expm1().round()
            q = exponents @ self.charges.to(exponents.dtype)
            compatible = (q[:,None,:] == q[None,:,:]).all(-1)
            d = (exponents[:,None,:]-exponents[None,:,:]).abs().sum(-1)
            similarity = (z@z.T).clamp(0,1) * torch.exp(-d/4) * compatible
            similarity.fill_diagonal_(0.)
            potential = float(similarity.max(-1).values.mean().clamp(0,1))
        env = torch.cat([F.one_hot(batch.n_point-4,3).to(C.dtype),C.new_tensor([[math.log1p(length)]]),
                         (batch.remaining_budget/batch.initial_budget).to(C.dtype).unsqueeze(-1),
                         batch.initial_budget.to(C.dtype).log1p().unsqueeze(-1)],-1)[0]
        extra=C.new_tensor([math.log1p(initial_terms),length/initial_terms,math.log1p(len(rows.action_indices)),
                            0.,math.log1p(len(selected.action_indices))])
        return SingleFeatures(C[0,:length].clone(),phi.clone(),left,right,
            (left*right).sum(-1)/self.actor.config.temperature,env,extra,selected,
            batch.context_ids[0],self.law_id,C.new_zeros(6),potential,batch.context_ids[0]).to("cpu")

    def forward(self, states):
        if not states or any(s.law_id!=self.law_id for s in states): raise ValueError("Wrong probability/context law")
        device=next(self.critic.parameters()).device; states=[s.to(device) for s in states]
        sizes=[len(s.context) for s in states]; C=torch.cat([s.context for s in states])
        phi=torch.cat([s.symbolic for s in states]); psi=torch.cat([s.actions.action_features for s in states]).to(C.dtype)
        U=self.actor.term_adapter(torch.cat([C,phi],-1)); G=torch.stack([u.mean(0) for u in U.split(sizes)])
        pooled=torch.stack([torch.cat([c.mean(0),p.mean(0)]) for c,p in zip(C.split(sizes),phi.split(sizes))])
        env=torch.stack([s.environment for s in states]); history=torch.stack([s.trajectory for s in states])
        value=self.critic(torch.cat([pooled,env,torch.stack([s.value_extra for s in states]),history],-1)).squeeze(-1)
        tables,offset,ptr,terms=[],0,[0],[]
        for b,(s,size) in enumerate(zip(states,sizes)):
            tables.append(replace(s.actions,action_batch=torch.full_like(s.actions.action_batch,b)))
            terms.append(s.actions.action_indices[:,0]+offset); offset+=size; ptr.append(ptr[-1]+len(s.base))
        table=ActionTable(**{f.name:torch.tensor(ptr,device=device) if f.name=='action_ptr' else torch.cat([getattr(t,f.name) for t in tables]) for f in fields(ActionTable)})
        left,right=torch.cat([s.left for s in states]),torch.cat([s.right for s in states])
        L,R=self.actor.bracket_adapter(left),self.actor.bracket_adapter(right)
        pair=torch.cat([L+R,(L-R).abs(),L*R],-1); b=table.action_batch
        joined=torch.cat([U[torch.cat(terms)],G[b],pair,psi,env[b],history[b]],-1)
        base=torch.cat([s.base for s in states]); logits=base+self.actor.action_scorer(joined).squeeze(-1)
        if not bool(torch.isfinite(value).all()): raise ValueError("Nonfinite critic")
        return CandidateOutput(logits,segmented_log_softmax(logits,table.action_ptr),base,table,
                               tuple(s.context_id for s in states),self.law_id),value
