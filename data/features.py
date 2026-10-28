"""Exact physical-configuration checks and expression reconstruction."""
import hashlib
import json
import math
from functools import reduce
from .symbolic import parse_terms
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':')).encode()).hexdigest()
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

def expression_from_tokens(tokens):
    import forsythia_hep as api
    from .symbolic import parse_terms
    result = 0 * api.abk(1, 2)
    for row in parse_terms(tokens):
        term = row.coefficient
        for factor in row.factors:
            constructor = api.abk if factor.kind == "angle" else api.sbk
            term = term * constructor(*factor.labels) ** factor.power
        result = result + term
    if result.tokenize() != tokens:
        raise ValueError("Exact token reconstruction changed order/content")
    return result
