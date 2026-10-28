"""Outcome-free action enumeration and network inputs. No engine/model imports."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import torch

REPO = Path(__file__).resolve().parents[2]
BRACKET = re.compile(r"(?:<([1-6])([1-6])>|\[([1-6])([1-6])\])")


class UnsupportedExpression(ValueError):
    pass


@lru_cache(maxsize=1)
def vocabulary():
    vocab = json.loads((REPO / "data/tokenizor.json").read_text(encoding="utf-8"))
    expected = [f"{left}{i}{j}{right}" for i in range(1, 6) for j in range(i + 1, 7)
                for left, right in (("<", ">"), ("[", "]"))]
    if any(vocab.get(token) != 16 + i for i, token in enumerate(expected)):
        raise ValueError("The 30 bracket IDs do not match this architecture")
    return vocab


def content_id(tokens):
    return hashlib.sha256(json.dumps(tokens, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Factor:
    kind: str
    labels: tuple[int, int]
    power: int
    token_id: int
    token_position: int


@dataclass(frozen=True)
class Term:
    coefficient: int
    factors: tuple[Factor, ...]
    metadata: int


def parse_terms(token_terms):
    """Current tokenizer grammar; reject unsupported syntax rather than skip it."""
    vocab = vocabulary()
    terms = []
    for tokens in token_terms:
        if (not isinstance(tokens, (tuple, list)) or len(tokens) < 5
                or tokens[0] not in ("+", "-") or tokens[-2] != "_"
                or tokens[-1] not in tuple(str(i) for i in range(7))
                or any(not isinstance(x, str) or x not in vocab for x in tokens)):
            raise UnsupportedExpression("Expected canonical signed integer monomial tokens with _ max_label")
        end, cursor = len(tokens) - 2, 1
        while cursor < end and len(tokens[cursor]) == 1 and tokens[cursor] in "0123456789":
            cursor += 1
        if cursor < 3:
            raise UnsupportedExpression("Coefficient requires at least two digit tokens")
        coefficient = int("".join(tokens[1:cursor])) * (1 if tokens[0] == "+" else -1)
        if coefficient == 0:
            raise UnsupportedExpression("Canonical zero must have no terms, not a zero-coefficient term")
        factors, seen = [], set()
        while cursor < end:
            token = tokens[cursor]
            match = BRACKET.fullmatch(token)
            if (match is None or cursor + 4 > end or tokens[cursor + 1] != "^"
                    or any(len(x) != 1 or x not in "0123456789" for x in tokens[cursor + 2:cursor + 4])):
                raise UnsupportedExpression("Only canonical brackets with positive two-digit powers are supported")
            labels = tuple(int(x) for x in match.groups() if x is not None)
            power = int("".join(tokens[cursor + 2:cursor + 4]))
            if labels[0] >= labels[1] or power == 0 or token in seen:
                raise UnsupportedExpression("Noncanonical bracket order, zero power or duplicate factor")
            seen.add(token)
            factors.append(Factor("angle" if token[0] == "<" else "square", labels, power, vocab[token], cursor))
            cursor += 4
        terms.append(Term(coefficient, tuple(factors), int(tokens[-1])))
    maximum = max((label for term in terms for f in term.factors for label in f.labels), default=0)
    if any(term.metadata != maximum for term in terms):
        raise UnsupportedExpression("_ metadata is the expression's maximum label, not its n_point")
    return tuple(terms)


def pair_is_legal(first, second, n_point):
    if type(n_point) is not int or n_point not in (4, 5, 6):
        raise ValueError("n_point must explicitly be 4, 5 or 6")
    if first.power <= 0 or second.power <= 0:
        return False
    if first.kind == second.kind:
        return len(set(first.labels + second.labels)) == 4
    return any(label <= n_point for label in set(first.labels) & set(second.labels))


def action_features(first, second):
    features = [0.] * 69
    for factor in (first, second):
        channel = factor.token_id - 16
        features[channel] += 1.
        features[30 + channel] += math.log1p(factor.power)
    combo = 0 if first.kind == second.kind == "angle" else 2 if first.kind == second.kind == "square" else 1
    features[60 + combo] = 1.
    for label in set(first.labels) & set(second.labels):
        features[63 + label - 1] = 1.
    return features


@dataclass(frozen=True)
class ActionTable:
    action_indices: torch.Tensor
    token_positions: torch.Tensor
    bracket_ids: torch.Tensor
    powers: torch.Tensor
    action_features: torch.Tensor
    action_batch: torch.Tensor
    action_ptr: torch.Tensor
    query_allowed: torch.Tensor
    source_rows: torch.Tensor

    def to(self, device):
        return replace(self, **{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def select(self, keep, batch_size):
        if keep.dtype != torch.bool or keep.shape != self.action_batch.shape:
            raise ValueError("Action selection mask mismatch")
        selected = {f.name: getattr(self, f.name)[keep] for f in fields(self) if f.name != "action_ptr"}
        sizes = torch.bincount(selected["action_batch"], minlength=batch_size)
        selected["action_ptr"] = torch.cat([sizes.new_zeros(1), sizes.cumsum(0)])
        return ActionTable(**selected)


@dataclass(frozen=True)
class PolicyInput:
    token_ids: torch.Tensor
    term_features: torch.Tensor
    actions: ActionTable
    n_point: torch.Tensor
    remaining_budget: torch.Tensor
    initial_budget: torch.Tensor
    state_ids: tuple[str, ...]
    context_ids: tuple[str, ...]

    def to(self, device):
        return replace(self, **{f.name: getattr(self, f.name).to(device)
                               for f in fields(self) if isinstance(getattr(self, f.name), (torch.Tensor, ActionTable))})


def build_policy_input(token_expressions, n_points, remaining_budget, initial_budget, tried_actions=None,
                       *, device="cpu", max_terms=128, max_tokens=128, max_actions=32768, max_batch=16):
    """Accept tokens only, preserve term/factor order, never query an outcome."""
    count = len(token_expressions)
    if not 1 <= count <= max_batch or min(max_terms, max_tokens, max_actions, max_batch) <= 0:
        raise ValueError("Invalid batch/capacity")
    if not all(len(x) == count for x in (n_points, remaining_budget, initial_budget)):
        raise ValueError("Particle/budget arrays must match batch size")
    tried_actions = [set() for _ in range(count)] if tried_actions is None else tried_actions
    if len(tried_actions) != count:
        raise ValueError("Query history must match batch size")
    parsed, snapshots, contexts, queried = [], [], [], []
    for tokens, n, remaining, initial, tried in zip(token_expressions, n_points, remaining_budget, initial_budget, tried_actions):
        if type(n) is not int or n not in (4, 5, 6):
            raise ValueError("n_point must be an explicit integer 4,5,6")
        if type(initial) is not int or type(remaining) is not int or not 0 <= remaining <= initial or initial <= 0:
            raise ValueError("Require integer budgets 0 <= remaining <= initial and initial > 0")
        if len(tokens) > max_terms or any(len(t) > max_tokens for t in tokens):
            raise UnsupportedExpression("Network capacity exceeded; no truncation")
        parsed.append(parse_terms(tokens))
        sid = content_id(tokens)
        snapshots.append(sid)
        normalized = set()
        for t, pair in tried:
            i, j = pair
            if any(type(v) is not int for v in (t, i, j)) or not 0 <= t < len(parsed[-1]) or not 0 <= i < j < len(parsed[-1][t].factors):
                raise ValueError("Query history contains stale/invalid action indices")
            if not pair_is_legal(parsed[-1][t].factors[i], parsed[-1][t].factors[j], n):
                raise ValueError("Query history contains a structurally illegal action")
            normalized.add((t, (i, j)))
        queried.append(normalized)
        contexts.append(content_id([sid, n, remaining, initial, sorted(normalized)]))
    T = max(1, max(map(len, parsed)))
    S = max(1, max((len(t) for expr in token_expressions for t in expr), default=0))
    ids = torch.zeros(count, T, S, dtype=torch.long)
    term_features = torch.zeros(count, T, 32)
    indices, positions, bracket_ids, powers, features, batch_ids, allowed, ptr = [], [], [], [], [], [], [], [0]
    vocab = vocabulary()
    for b, (tokens, terms, n) in enumerate(zip(token_expressions, parsed, n_points)):
        logs = [math.log(abs(term.coefficient)) for term in terms]
        mean_log = sum(logs) / len(logs) if logs else 0.
        for t, (raw, term) in enumerate(zip(tokens, terms)):
            ids[b, t, :len(raw)] = torch.tensor([vocab[x] for x in raw])
            for factor in term.factors:
                term_features[b, t, factor.token_id - 16] = math.log1p(factor.power)
            term_features[b, t, 30] = 1 if term.coefficient > 0 else -1
            term_features[b, t, 31] = logs[t] - mean_log
            for i, j in itertools.combinations(range(len(term.factors)), 2):
                first, second = term.factors[i], term.factors[j]
                if not pair_is_legal(first, second, n):
                    continue
                if len(indices) - ptr[-1] >= max_actions:
                    raise UnsupportedExpression("Action capacity exceeded; no silent top-k filtering")
                indices.append((t, i, j)); positions.append((first.token_position, second.token_position))
                bracket_ids.append((first.token_id, second.token_id)); powers.append((first.power, second.power))
                features.append(action_features(first, second)); batch_ids.append(b)
                allowed.append(remaining_budget[b] > 0 and (t, (i, j)) not in queried[b])
        ptr.append(len(indices))
    table = ActionTable(torch.tensor(indices, dtype=torch.long).reshape(-1, 3),
                        torch.tensor(positions, dtype=torch.long).reshape(-1, 2),
                        torch.tensor(bracket_ids, dtype=torch.long).reshape(-1, 2),
                        torch.tensor(powers, dtype=torch.long).reshape(-1, 2),
                        torch.tensor(features, dtype=torch.float32).reshape(-1, 69),
                        torch.tensor(batch_ids, dtype=torch.long), torch.tensor(ptr, dtype=torch.long),
                        torch.tensor(allowed, dtype=torch.bool), torch.arange(len(indices)))
    return PolicyInput(ids, term_features, table, torch.tensor(n_points), torch.tensor(remaining_budget),
                       torch.tensor(initial_budget), tuple(snapshots), tuple(contexts)).to(device)
