"""Opt-in checkpoint-growth punishment on the frozen 20-step PPO pipeline.

No experiment is launched on import. Historical runners, network inputs,
sampling and the three original reward components remain unchanged.
"""
from dataclasses import asdict, dataclass, replace
import math

from .lambda20_experiment import Runtime20, collect20, seed_for
from .lambda20_policy import CUTS, HORIZON


def validate_coefficient(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("c_punishment must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError("c_punishment must be a finite nonnegative number")
    return float(value)


@dataclass(frozen=True)
class CheckpointPunishmentState:
    initial_terms: int
    previous_checkpoint_terms: int
    previous_checkpoint_step: int = 0
    last_step: int = 0
    last_terms: int = 0
    terminal: bool = False

    def __post_init__(self):
        counts = (self.initial_terms, self.previous_checkpoint_terms,
                  self.previous_checkpoint_step, self.last_step, self.last_terms)
        if any(type(x) is not int for x in counts) or type(self.terminal) is not bool:
            raise ValueError("Expected integer lengths/steps and Boolean termination")
        if self.initial_terms <= 0 or min(counts) < 0 or self.last_step > HORIZON:
            raise ValueError("Invalid checkpoint punishment state")
        expected = self.last_step if self.terminal else max((0, *(t for t in CUTS if t <= self.last_step)))
        if self.previous_checkpoint_step != expected:
            raise ValueError("Missing or inconsistent checkpoint history")
        if self.previous_checkpoint_step == 0 and self.previous_checkpoint_terms != self.initial_terms:
            raise ValueError("The first checkpoint must compare with initial length")
        if self.last_step == 0 and (self.last_terms != self.initial_terms or
                                   self.previous_checkpoint_terms != self.initial_terms):
            raise ValueError("Initial checkpoint must refer to the initial expression")
        if self.previous_checkpoint_step == self.last_step and self.previous_checkpoint_terms != self.last_terms:
            raise ValueError("Checkpoint length differs from actual length")
        if not self.terminal and (self.last_step == HORIZON or self.last_terms == 0):
            raise ValueError("Horizon/zero expression must terminate")
        if not self.terminal and self.previous_checkpoint_terms == 0:
            raise ValueError("Cannot continue an episode after a zero-expression checkpoint")

    @classmethod
    def start(cls, initial_terms, *, terminal=False):
        return cls(initial_terms, initial_terms, last_terms=initial_terms, terminal=terminal)

    def advance(self, step, actual_terms, terminal, *, c_punishment=0.):
        coefficient = validate_coefficient(c_punishment)
        if type(step) is not int or step != self.last_step + 1 or self.terminal:
            raise ValueError("Advance each real transition exactly once, in episode order")
        if type(actual_terms) is not int or actual_terms < 0 or type(terminal) is not bool:
            raise ValueError("Invalid actual successor length/termination")
        if step > HORIZON or (step == HORIZON or actual_terms == 0) and not terminal:
            raise ValueError("Invalid terminal boundary")
        settle = step in CUTS or terminal
        growth = max(actual_terms - self.previous_checkpoint_terms, 0) if settle else 0
        weight = step / HORIZON if settle else 0.
        magnitude = weight * (growth / self.initial_terms)
        contribution = -coefficient * magnitude
        if not math.isfinite(contribution):
            raise ValueError("Nonfinite weighted punishment")
        after = replace(self, last_step=step, last_terms=actual_terms, terminal=terminal,
                        previous_checkpoint_terms=actual_terms if settle else self.previous_checkpoint_terms,
                        previous_checkpoint_step=step if settle else self.previous_checkpoint_step)
        return after, {
            "settled": settle, "episode_step": step, "horizon": HORIZON,
            "initial_terms": self.initial_terms,
            "previous_checkpoint_step": self.previous_checkpoint_step,
            "previous_checkpoint_terms": self.previous_checkpoint_terms,
            "actual_terms": actual_terms, "growth": growth, "time_weight": weight,
            "punishment": magnitude, "c_punishment": coefficient,
            "weighted_punishment": contribution,
        }


class PunishmentRuntime20(Runtime20):
    """The same environment transitions, with an optional fourth reward term.

    c_punishment=0 bypasses all new history/reward fields and returns exactly
    the historical transition records. Construction itself requires opt-in.
    """
    def __init__(self, out, model, ledger, rewards, gae_lambda=.94, *,
                 c_punishment=0., enabled=False):
        if not enabled:
            raise RuntimeError("Checkpoint punishment training interface is off")
        self.c_punishment = validate_coefficient(c_punishment)
        if not math.isfinite(gae_lambda) or not 0 <= gae_lambda <= 1:
            raise ValueError("Invalid GAE lambda")
        if model.top_k != 16:
            raise ValueError("This adapter retains Runtime20's fixed top16 sampler")
        super().__init__(out, model, ledger, rewards, gae_lambda)

    def start(self, source):
        episode = super().start(source)
        if self.c_punishment:
            episode["checkpoint_punishment"] = CheckpointPunishmentState.start(
                episode["progress"].initial, terminal=episode["state"] is None)
        return episode

    @staticmethod
    def _check_history(episode, history):
        if not isinstance(history, CheckpointPunishmentState):
            raise ValueError("Missing checkpoint history; cannot infer it from best length")
        progress = episode["progress"]
        if (history.initial_terms != progress.initial or history.last_step != progress.step or
                history.last_terms != len(episode["expr"].tokenize()) or
                history.terminal != (episode["state"] is None)):
            raise ValueError("Checkpoint history does not match the restored episode")

    def transition(self, episode, generator):
        if self.model.top_k != 16:
            raise ValueError("Model and fixed Runtime20 sampler must both use top16")
        if not self.c_punishment:
            return super().transition(episode, generator)
        history = episode.get("checkpoint_punishment")
        self._check_history(episode, history)  # Fail before executing an action.
        state, row = super().transition(episode, generator)
        after, details = history.advance(row["step"], row["after"], row["terminal"],
                                         c_punishment=self.c_punishment)
        self._check_history(episode, after)
        reward = dict(row["reward"])
        reward.update(punishment=details["punishment"],
                      weighted_punishment=details["weighted_punishment"],
                      total=reward["total"] + details["weighted_punishment"])
        if not math.isfinite(reward["total"]):
            raise ValueError("Nonfinite total reward")
        episode["checkpoint_punishment"] = after
        return state, dict(row, reward=reward, checkpoint_punishment=details)

    def punishment_state_dict(self):
        """Save alongside the existing model/optimizer/deck/lane recovery data."""
        result = {"version": 1, "c_punishment": self.c_punishment, "lanes": None}
        if not self.c_punishment:
            return result
        lanes = []
        for episode in self.lanes:
            if episode is None:
                lanes.append(None)
                continue
            history = episode.get("checkpoint_punishment")
            self._check_history(episode, history)
            lanes.append({"episode": episode["episode"], "source_id": episode["source"]["source_id"],
                          "progress": asdict(episode["progress"]), "state": asdict(history)})
        result["lanes"] = lanes
        return result

    def load_punishment_state_dict(self, saved):
        """Restore after restoring lanes; reject mismatches before mutating any lane."""
        if not isinstance(saved, dict) or set(saved) != {"version", "c_punishment", "lanes"}:
            raise ValueError("Missing punishment recovery metadata")
        if saved["version"] != 1 or validate_coefficient(saved["c_punishment"]) != self.c_punishment:
            raise ValueError("Punishment recovery version/coefficient mismatch")
        if not self.c_punishment:
            if saved["lanes"] is not None:
                raise ValueError("Unexpected history for disabled punishment")
            return
        records = saved["lanes"]
        if not isinstance(records, list) or len(records) != len(self.lanes):
            raise ValueError("Punishment recovery lane count mismatch")
        restored = []
        for episode, record in zip(self.lanes, records):
            if episode is None or record is None:
                if episode is not None or record is not None:
                    raise ValueError("Punishment recovery lane occupancy mismatch")
                restored.append(None)
                continue
            if (not isinstance(record, dict) or set(record) != {"episode", "source_id", "progress", "state"} or
                    record["episode"] != episode["episode"] or
                    record["source_id"] != episode["source"]["source_id"] or
                    record["progress"] != asdict(episode["progress"])):
                raise ValueError("Punishment recovery episode mismatch")
            try:
                history = CheckpointPunishmentState(**record["state"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid saved checkpoint history") from exc
            self._check_history(episode, history)
            restored.append(history)
        for episode, history in zip(self.lanes, restored):
            if episode is not None:
                episode["checkpoint_punishment"] = history


def train_ppo_round(runtime, deck, optimizer, round_index, *, on_rollout, on_step,
                    c_entropy=.001, enabled=False):
    """Collect actual penalized rewards -> per-lane GAE -> 60 entropy-PPO updates.

    Caller owns the unique experiment ID, frozen budget, durable logging and
    complete recovery checkpoint. This is not a new experiment CLI or launcher.
    """
    if not enabled:
        raise RuntimeError("Checkpoint punishment PPO round is off")
    if not isinstance(runtime, PunishmentRuntime20):
        raise TypeError("Expected PunishmentRuntime20")
    if (type(round_index) is not int or not 1 <= round_index <= 36 or
            round_index != runtime.round + 1 or runtime.phase != "train"):
        raise ValueError("Expected the next training round in the fixed 36-round budget")
    if runtime.model.top_k != 16 or not callable(on_rollout) or not callable(on_step):
        raise ValueError("Expected top16 and explicit rollout/update logging callbacks")
    if not math.isfinite(c_entropy) or c_entropy < 0:
        raise ValueError("Invalid entropy coefficient")
    import torch
    from .entropy20_policy import update_ppo480_entropy

    runtime.frozen_check()
    # collect20 consumes row['reward']['total'] after transition has deducted
    # punishment, so BOTH GAE and critic targets include the new component.
    states, buffer = collect20(runtime, deck, round_index)
    buffer["c_punishment"] = runtime.c_punishment
    buffer["checkpoint_punishment_state"] = runtime.punishment_state_dict()
    on_rollout(buffer)  # Persist before any optimizer step.
    updates = update_ppo480_entropy(
        runtime.model, optimizer, states, buffer["rows"], buffer["advantages"], buffer["targets"],
        generator=torch.Generator().manual_seed(seed_for("update", round_index)),
        hit=runtime.ledger.hit, on_step=on_step, enabled=True, entropy_coef=c_entropy)
    if len(updates) != 60:
        raise ValueError("Expected exactly 60 PPO optimizer steps")
    runtime.frozen_check()
    return buffer, updates
