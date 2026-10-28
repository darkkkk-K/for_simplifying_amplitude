"""One probability implementation for sequential without-replacement actions."""
from dataclasses import asdict, dataclass
import math

import torch

from .legal_actions import content_id


def ordered_log_prob(logits, selected):
    values = logits.double()
    if values.ndim != 1 or not bool(torch.isfinite(values).all()) or not selected:
        raise ValueError("Finite logits and at least one action required")
    keep = torch.ones(len(values), dtype=torch.bool, device=values.device)
    result = []
    for index in selected:
        if type(index) is not int or not 0 <= index < len(values) or not bool(keep[index]):
            raise ValueError("Repeated/out-of-range selected action")
        result.append(values[index] - torch.logsumexp(values[keep.clone()], 0))
        keep[index] = False
    return torch.stack(result)


@dataclass(frozen=True)
class DecisionRecord:
    law_id: str
    context_id: str
    support_id: str
    selected_actions: tuple
    conditional_log_probs: tuple
    sampling: str = "ordered_without_replacement"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(value["law_id"], value["context_id"], value["support_id"],
                   tuple((int(t), (int(i), int(j))) for t, (i, j) in value["selected_actions"]),
                   tuple(float(x) for x in value["conditional_log_probs"]), value["sampling"])


def draw_actions(output, batch_index, count, generator):
    if type(count) is not int or count <= 0 or generator is None or str(generator.device) != "cpu":
        raise ValueError("Positive count and explicit CPU generator required")
    start, end = output.actions.action_ptr[batch_index:batch_index + 2].detach().cpu().tolist()
    actions = output.actions_for(batch_index)
    if not actions:
        raise ValueError("Empty action list is terminal, not a uniform distribution")
    # CPU float64 sampling; do not accidentally retain rollout autograd graphs.
    logits = output.logits[start:end].detach().cpu().double()
    available, chosen = list(range(len(actions))), []
    for _ in range(min(count, len(actions))):
        local = int(torch.multinomial(logits[available].softmax(0), 1, generator=generator).item())
        chosen.append(available.pop(local))
    logs = ordered_log_prob(logits, chosen).tolist()
    return DecisionRecord(output.law_id, output.context_ids[batch_index], content_id(actions),
                          tuple(actions[i] for i in chosen), tuple(logs))


def recompute_log_prob(output, batch_index, record):
    actions = output.actions_for(batch_index)
    if (record.sampling != "ordered_without_replacement" or record.law_id != output.law_id
            or record.context_id != output.context_ids[batch_index] or record.support_id != content_id(actions)):
        raise ValueError("State/budget/history/support/probability rule changed")
    lookup = {action: i for i, action in enumerate(actions)}
    if (len(record.selected_actions) != len(record.conditional_log_probs)
            or any(not math.isfinite(x) or x > 1e-10 for x in record.conditional_log_probs)):
        raise ValueError("Malformed behavior record")
    try:
        selected = [lookup[action] for action in record.selected_actions]
    except KeyError as exc:
        raise ValueError("Selected action no longer in support") from exc
    start, end = output.actions.action_ptr[batch_index:batch_index + 2].detach().cpu().tolist()
    return ordered_log_prob(output.logits[start:end], selected)


def policy_gradient_loss(output, batch_index, record, advantage):
    """REINFORCE loss for one ordered candidate set; caller supplies trajectory advantage."""
    conditional = recompute_log_prob(output, batch_index, record)
    old = torch.tensor(record.conditional_log_probs, dtype=conditional.dtype, device=conditional.device)
    if not torch.allclose(conditional.detach(), old, atol=1e-6, rtol=0.):
        raise ValueError("On-policy loss refuses stale behavior probabilities; no implicit off-policy update")
    logp = conditional.sum()
    value = torch.as_tensor(advantage, device=logp.device, dtype=logp.dtype).detach()
    if value.numel() != 1 or not bool(torch.isfinite(value)):
        raise ValueError("One finite detached trajectory advantage required")
    return -value.reshape(()) * logp
