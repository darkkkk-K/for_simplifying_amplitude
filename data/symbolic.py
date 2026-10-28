"""Canonical brackets, integer monomials, and legal identity pairs."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
REPO = Path(__file__).resolve().parents[1]
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
