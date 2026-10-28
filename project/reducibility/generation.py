"""Bounded whole-expression generation and weak-negative annealing labels."""
from collections import Counter
import gzip
import hashlib
import json
import math
import random
import time
import numpy as np
from .common import (ROOT, Ledger, BudgetExceeded, digest, save_json, file_hash,
                     paths, check_protocol, storage_bytes)
from .features import validate, expression_key, action_count
from .storage import write_shards


def amplitude_config(point, dimension, rng, nrng):
    """Adapted from data/generation.py; explicit RNG, no import-time side effects."""
    if point not in (4, 5, 6) or dimension < 2:
        raise ValueError('Unsupported configuration')
    if dimension % 2 == 0:
        total = 4 * rng.randint(0, min(dimension // 2, 5 * point // 4))
    else:
        total = 4 * rng.randint(0, min((dimension - 1) // 2, math.ceil((5 * point - 4) / 4))) + 2
    values = nrng.multinomial(total, [1 / point] * point)
    excessive = np.flatnonzero(values > 5)
    excess = int(sum(values[excessive] - 5))
    for _ in range(excess):
        values[np.argmin(values)] += 1
    values[excessive] = 5
    signs = 2 * nrng.integers(0, 2, point) - 1
    hels = (values * signs).tolist()
    if -sum(h for h in hels if h < 0) % 2:
        i = next(i for i, h in enumerate(hels) if h < 0 and h % 2)
        hels[i] = -hels[i]
    absolute = sum(abs(h) for h in hels)
    if absolute % 2 or dimension < absolute // 2 or (
            sum(max(-h, 0) for h in hels) + dimension - absolute // 2) % 2:
        raise ValueError('Generated invalid configuration')
    return hels


def quota_counts(total, probabilities):
    exact = np.asarray(probabilities) * total
    result = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - result), kind='stable')[:total - sum(result)]:
        result[i] += 1
    return result.tolist()


def gap_bin(gap):
    return 0 if gap == 1 else 1 if gap == 2 else 2 if gap <= 5 else 3


class Engine:
    def __init__(self, ledger, config):
        import forsythia_hep as api
        self.api, self.ledger, self.config = api, ledger, config

    def normalize(self, expr):
        self.ledger.add('normal_order_calls')
        expr = expr.normal_order()
        self.ledger.add('expand_calls')
        return expr.expand()

    def tokens(self, expr):
        self.ledger.add('tokenize_calls')
        return expr.tokenize()

    def expression(self, count, hels, dim, rng):
        terms = []
        for _ in range(count):
            self.ledger.add('monomial_calls')
            mono = self.api.random_monomial(hels, dim)
            tok = self.tokens(mono)
            if not tok:
                self.ledger.add('zero_monomials')
                return None
            validate(tok, len(hels), dim, hels)
            terms.append(rng.choice((-3, -2, -1, 1, 2, 3)) * mono)
        self.ledger.add('expression_constructions')
        return self.normalize(sum(terms))

    def scramble(self, expr, count, n, rng, dim, hels):
        changed = 0
        for _ in range(count):
            self.ledger.add('scramble_calls')
            if rng.random() < 0.5:
                result = self.api.random_apply_schouten(expr, 1)[1]
                self.ledger.add('schouten_calls')
            else:
                result = self.api.random_apply_momentum_conservation(expr, n, 1)[1]
                self.ledger.add('momentum_calls')
            result = self.normalize(result)
            changed += int(result != expr)
            expr = result
            tokens = self.tokens(expr)
            validate(tokens, n, dim, hels)
            if len(tokens) > self.config['intermediate_terms_max'] or max(map(len, tokens), default=0) > 128:
                return None, changed
        return expr, changed

    def screen(self, expr, n, dim, hels):
        tokens = self.tokens(expr)
        initial = len(tokens)
        t0 = (2 * initial + 1) / -math.log(self.config['anneal_initial_expand_probability'])
        summaries = []
        for repeat in range(self.config['anneal_repeats']):
            self.ledger.add('anneal_calls')
            accepted, history, final = self.api.simplify(
                expr, n_point=n, n_max_times=self.config['anneal_steps'], temperature=t0,
                temperature_end=0.0, sample_nth_root=-1.0, record_all_history=True)
            if accepted != len(history) or accepted > self.config['anneal_steps']:
                raise RuntimeError('Unexpected annealing return/history contract')
            lowest, lowest_tokens = initial, tokens
            lengths = [initial]
            for state in [*[item[1] for item in history], final]:
                st = self.tokens(state)
                validate(st, n, dim, hels)
                if len(st) > 128 or max(map(len, st), default=0) > 128:
                    raise RuntimeError('Annealing history capacity exceeded')
                lengths.append(len(st))
                if len(st) < lowest:
                    lowest, lowest_tokens = len(st), st
            self.ledger.add('anneal_accepted_steps', accepted)
            self.ledger.add('anneal_outer_iteration_upper', self.config['anneal_steps'])
            self.ledger.add('anneal_internal_attempt_upper', 42 * self.config['anneal_steps'])
            summaries.append({'repeat': repeat, 'initial_temperature': t0, 'accepted_steps': accepted,
                              'minimum_terms': lowest, 'final_terms': lengths[-1],
                              'visited_lengths': lengths, 'minimum_tokens': lowest_tokens})
        return summaries


def screened_negative(summaries, initial, repeats=3):
    if len(summaries) != repeats:
        raise ValueError('Incomplete screening cannot label a negative')
    return all(r['minimum_terms'] >= initial for r in summaries)


def load_exclusions(config, pilot_id=None):
    keys, legacy_sources, file_digests = set(), set(), {}
    for rel in config['exclusion_files']:
        path = ROOT / rel
        records = json.loads(path.read_text(encoding='utf-8'))
        file_digests[rel] = file_hash(path)
        for r in records:
            keys.add(expression_key(r['tokens'], int(r['n_point'])))
            if r.get('source_id'):
                legacy_sources.add(r['source_id'])
    if pilot_id:
        pilot = ROOT / 'datasets' / pilot_id / 'seen_keys.json'
        keys.update(json.loads(pilot.read_text(encoding='utf-8')))
        file_digests[pilot.relative_to(ROOT).as_posix()] = file_hash(pilot)
    return keys, legacy_sources, file_digests


def legacy_hashes(tokens):
    return {hashlib.sha256(repr(tokens).encode()).hexdigest(),
            hashlib.sha256(json.dumps(tokens, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()}


def run(run_id):
    protocol = check_protocol(run_id, 'generation')
    cfg, mode = protocol['config'], protocol['extra']['mode']
    setup = cfg[mode]
    out = paths(run_id)[1]
    dataset = ROOT / 'datasets' / run_id
    if dataset.exists():
        raise FileExistsError('Dataset ID already exists')
    dataset.mkdir(parents=True)
    ledger = Ledger(setup['limits'])
    engine = Engine(ledger, cfg)
    rng = random.Random(cfg['data_seed'] + (0 if mode == 'pilot' else 1))
    nrng = np.random.default_rng(cfg['data_seed'] + (0 if mode == 'pilot' else 1))
    seen, legacy_sources, exclusion_hashes = load_exclusions(cfg, protocol['extra'].get('pilot_id'))
    if exclusion_hashes != protocol['extra']['exclusion_hashes']:
        raise RuntimeError('Exclusion data changed after freeze')
    original_seen = set(seen)
    slots, quotas, gaps = [], {}, Counter()
    for split, count in setup['split_pairs'].items():
        point_counts = quota_counts(count, cfg['point_probabilities'])
        slots.extend((split, point) for point, k in zip(cfg['points'], point_counts) for _ in range(k))
        quotas[split] = quota_counts(count, cfg['gap_probabilities'])
    rng.shuffle(slots)
    rows, evidence, auxiliary, audit = [], [], [], []
    rejections = Counter()
    started, status, error = time.time(), 'incomplete', None
    pairs = 0
    audit_queue = []
    try:
        while slots:
            split, point = slots[-1]
            ledger.add('positive_candidates')
            ledger.add('config_calls')
            dim = rng.randint(cfg['dimension_min'], cfg['dimension_max'])
            hels = amplitude_config(point, dim, rng, nrng)
            source = engine.expression(rng.randint(1, cfg['source_terms_max']), hels, dim, rng)
            if source is None:
                rejections['zero_source_monomial'] += 1
                continue
            src_tokens = engine.tokens(source)
            if not 1 <= len(src_tokens) <= cfg['source_terms_max']:
                rejections['source_length'] += 1
                continue
            key_src = expression_key(src_tokens, point)
            if key_src in seen or legacy_hashes(src_tokens) & legacy_sources:
                rejections['duplicate_source'] += 1
                continue
            seen.add(key_src)
            steps = rng.choices(cfg['scramble_steps'], cfg['scramble_probabilities'])[0]
            positive, changed = engine.scramble(source, steps, point, rng, dim, hels)
            if positive is None:
                rejections['intermediate_capacity'] += 1
                continue
            pt = engine.tokens(positive)
            gap = len(pt) - len(src_tokens)
            if gap <= 0 or len(pt) > cfg['input_terms_max']:
                rejections['no_growth_or_capacity'] += 1
                continue
            bin_index = gap_bin(gap)
            if gaps[(split, bin_index)] >= quotas[split][bin_index]:
                rejections['gap_quota_full'] += 1
                continue
            key_pos = expression_key(pt, point)
            if key_pos in seen or legacy_hashes(pt) & legacy_sources:
                rejections['duplicate_positive'] += 1
                continue
            seen.add(key_pos)
            group = f'{run_id}_{ledger.counts["positive_candidates"]:06d}'
            negative, nt, results = None, None, None
            for _ in range(cfg['pair_attempts_max']):
                ledger.add('negative_candidates')
                candidate = engine.expression(len(pt), hels, dim, rng)
                if candidate is None:
                    rejections['zero_negative_monomial'] += 1
                    continue
                ct = engine.tokens(candidate)
                if len(ct) != len(pt):
                    rejections['negative_length_after_combining'] += 1
                    continue
                ck = expression_key(ct, point)
                if ck in seen or legacy_hashes(ct) & legacy_sources:
                    rejections['duplicate_negative'] += 1
                    continue
                seen.add(ck)
                screened = engine.screen(candidate, point, dim, hels)
                if screened_negative(screened, len(ct)):
                    negative, nt, results = candidate, ct, screened
                    break
                auxiliary.append({'group_id': group, 'split': split, 'n_point': point,
                                  'tokens': ct, 'label': 1, 'screen': screened})
                rejections['random_candidate_reducible'] += 1
            if negative is None:
                rejections['pair_failed'] += 1
                continue
            base = {'group_id': group, 'split': split, 'n_point': point, 'dimension': dim,
                    'helicities_twice': hels, 'term_count': len(pt)}
            rows.extend([{**base, 'id': group + '_positive', 'tokens': pt, 'label': 1,
                          'label_method': 'shorter_equivalent_witness', 'evidence_id': group},
                         {**base, 'id': group + '_negative', 'tokens': nt, 'label': 0,
                          'label_method': 'annealing_3x16_no_improvement', 'evidence_id': group}])
            evidence.append({'group_id': group, 'source_tokens': src_tokens, 'source_terms': len(src_tokens),
                             'term_gap': gap, 'scramble_calls': steps, 'changed_calls': changed,
                             'negative_screen': results, 'positive_actions': action_count(pt, point),
                             'negative_actions': action_count(nt, point)})
            if mode == 'pilot' or split == 'train':
                audit_queue.append((positive, point, dim, hels, group, gap))
            gaps[(split, bin_index)] += 1
            slots.pop()
            pairs += 1
            if pairs % 32 == 0:
                print(json.dumps({'pairs': pairs, 'remaining': len(slots), 'counts': ledger.counts,
                                  'seconds': round(time.time() - started, 1)}), flush=True)
        rng.shuffle(audit_queue)
        for expr, n, d, h, group, gap in audit_queue[:setup['positive_audit']]:
            screened = engine.screen(expr, n, d, h)
            audit.append({'group_id': group, 'term_gap': gap, 'n_point': n,
                          'missed': screened_negative(screened, len(engine.tokens(expr))), 'screen': screened})
        status = 'completed'
    except BudgetExceeded as exc:
        error = 'Budget exhausted: ' + str(exc)
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        status = 'failed'
    finally:
        for split in setup['split_pairs']:
            records = [r for r in rows if r['split'] == split]
            write_shards(dataset, split, records)
        for name, records in [('evidence', evidence), ('discovered_positives', auxiliary), ('positive_audit', audit)]:
            with gzip.open(dataset / f'{name}.jsonl.gz', 'wt', encoding='utf-8') as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=True) + '\n')
        save_json(dataset / 'seen_keys.json', sorted(seen - original_seen))
        hard = [a for a in audit if a['term_gap'] <= 2]
        hard_miss = sum(a['missed'] for a in hard) / len(hard) if hard else None
        summary = {'status': status, 'error': error, 'pairs': pairs, 'rows': len(rows),
                   'counts': ledger.counts, 'rejections': dict(rejections),
                   'split_counts': dict(Counter(r['split'] for r in rows)),
                   'positive_gap_histogram': dict(Counter(e['term_gap'] for e in evidence)),
                   'term_histogram': dict(Counter(r['term_count'] for r in rows)),
                   'point_histogram': dict(Counter(r['n_point'] for r in rows)),
                   'positive_audit_count': len(audit), 'near_positive_audit_count': len(hard),
                   'near_positive_annealing_miss_rate': hard_miss,
                   'pilot_quality_pass': status == 'completed' and hard_miss is not None and hard_miss <= 0.10,
                   'seconds': time.time() - started, 'files': {p.name: file_hash(p) for p in dataset.iterdir()},
                   'exclusion_hashes': exclusion_hashes, 'engine': protocol['engine']}
        if storage_bytes(dataset) > setup['storage_bytes']:
            summary['status'], summary['error'] = 'failed', 'Storage limit exceeded'
        save_json(dataset / 'manifest.json', summary)
        save_json(out / 'summary.json', summary)
        print(json.dumps(summary, ensure_ascii=True), flush=True)
    return summary
