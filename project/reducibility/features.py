"""Exact structural checks and 33-dimensional, target-free features."""
import math
from functools import reduce
from itertools import combinations
import torch
from project.rl.legal_actions import parse_terms, pair_is_legal
from .common import digest


def structures(tokens, n_point):
    if type(n_point) is not int or n_point not in (4, 5, 6):
        raise ValueError('Explicit n_point 4/5/6 required')
    terms = parse_terms(tokens)
    rows, dimensions, weights = [], [], []
    for term in terms:
        exponents, w = [0] * 30, [0] * n_point
        for f in term.factors:
            if max(f.labels) > n_point:
                raise ValueError('Bracket outside particle labels')
            exponents[f.token_id - 16] = f.power
            for i in f.labels:
                w[i - 1] += f.power * (1 if f.kind == 'angle' else -1)
        rows.append((term.coefficient, tuple(exponents)))
        dimensions.append(sum(exponents))
        weights.append(tuple(w))
    if len(set(dimensions)) > 1 or len(set(weights)) > 1:
        raise ValueError('Terms have different dimensions or helicities')
    if len({e for _, e in rows}) != len(rows):
        raise ValueError('Input must have like terms combined')
    return terms, rows, dimensions, weights


def validate(tokens, n_point, dimension, helicities):
    _, _, dims, weights = structures(tokens, n_point)
    if dims and (dims[0] != dimension or weights[0] != tuple(-h for h in helicities)):
        raise ValueError('Expression violates generating physical configuration')


def expression_key(tokens, n_point, scale_invariant=True):
    _, rows, _, _ = structures(tokens, n_point)
    rows.sort(key=lambda row: row[1])
    if rows and scale_invariant:
        divisor = reduce(math.gcd, (abs(c) for c, _ in rows)) * (1 if rows[0][0] > 0 else -1)
        rows = [(c // divisor, e) for c, e in rows]
    return digest([n_point, rows])


def make_features(tokens, n_point, max_terms=32):
    terms, rows, _, _ = structures(tokens, n_point)
    if not 1 <= len(terms) <= max_terms:
        raise ValueError('Empty or over-capacity classification input')
    log_c = [math.log(abs(c)) for c, _ in rows]
    mean_log = sum(log_c) / len(log_c)
    return torch.tensor([[*[math.log1p(v) for v in e], float(c > 0) * 2 - 1,
                          log_c[t] - mean_log, (n_point - 4) / 2]
                         for t, (c, e) in enumerate(rows)], dtype=torch.float32)


def action_count(tokens, n_point):
    return sum(pair_is_legal(a, b, n_point) for t in parse_terms(tokens)
               for a, b in combinations(t.factors, 2))


def collate(items):
    if not items:
        raise ValueError('Empty batch')
    x = torch.zeros(len(items), max(len(a[0]) for a in items), 33)
    mask = torch.zeros(x.shape[:2], dtype=torch.bool)
    for i, (feature, _) in enumerate(items):
        x[i, :len(feature)] = feature
        mask[i, :len(feature)] = True
    return x, mask, torch.tensor([y for _, y in items], dtype=torch.long)
