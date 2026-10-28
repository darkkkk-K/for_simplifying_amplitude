"""One frozen generation/evaluation protocol using PolicySimplifier unchanged."""
from collections import Counter
from functools import lru_cache
import ast
import gzip
import hashlib
import importlib.util
from itertools import combinations, combinations_with_replacement
import json
import os
from pathlib import Path
import random
import shutil
import statistics as st
import time

SOURCE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('AITOMMA_PORTFOLIO_ROOT', str(SOURCE_ROOT))).resolve()
ID = 'portfolio_fourpoint1024_20261028_v1'
CONTRACT = ROOT/'experiments'/ID
OUT = ROOT/'artifacts/runs'/ID
DATA = ROOT/'datasets'/ID
CONFIG = ROOT/'configs'/(ID+'.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, allow_nan=False)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def content(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


class Counts:
    def __init__(self, caps):
        self.caps, self.values = caps, Counter()

    def hit(self, name, amount=1):
        if name not in self.caps or self.values[name]+amount > self.caps[name]:
            raise RuntimeError('Budget exceeded: '+name)
        self.values[name] += amount


@lru_cache(maxsize=9)
def monomial_capacities(dimension):
    """Exact count by little-group weight before applying algebra identities."""
    brackets = [(sign, i, j) for sign in (1, -1) for i, j in combinations(range(4), 2)]
    counts = Counter()
    for factors in combinations_with_replacement(range(12), dimension):
        weights = [0]*4
        for factor in factors:
            sign, i, j = brackets[factor]
            weights[i] += sign
            weights[j] += sign
        counts[tuple(-v for v in weights)] += 1
    return counts


def group_name(dimension):
    if 2 <= dimension <= 4:
        return '2-4'
    if 5 <= dimension <= 7:
        return '5-7'
    if 8 <= dimension <= 10:
        return '8-10'
    raise ValueError('Dimension outside showcase scope')


def statistics(rows):
    if not rows:
        raise ValueError('An evaluation group is empty')
    initial = sum(r['initial_terms'] for r in rows)
    returned = sum(r['best_terms'] for r in rows)
    return dict(count=len(rows), mean_reduction=st.fmean((r['initial_terms']-r['best_terms'])/r['initial_terms'] for r in rows),
                total_term_reduction=1-returned/initial, terminal_mean_reduction=st.fmean((r['initial_terms']-r['final_terms'])/r['initial_terms'] for r in rows),
                mean_input_terms=initial/len(rows), mean_output_terms=returned/len(rows),
                initial_terms_total=initial, returned_terms_total=returned,
                mean_steps=st.fmean(r['steps'] for r in rows),
                improved_sources=sum(r['best_terms'] < r['initial_terms'] for r in rows))


def dependency_files():
    """Static local import closure, plus definitions loaded through the AST loader."""
    seeds = ['scripts/run_portfolio_evaluation.py', 'project/showcase/evaluation.py', 'scripts/simplify_policy.py',
             'project/rl/point5_curriculum_data.py', 'project/rl/packed32_common.py',
             'scripts/pretrain.py', 'scripts/generate_data.py', 'scripts/visualize_tsne_ex.py',
             'scripts/visualize_tsne_dim.py', 'scripts/visualize_tsne_sim.py',
             'project/rl/entropy20_policy.py', 'project/rl/restricted_runtime.py']
    seeds += [f'models/{name}.py' for name in ('encoder', 'custom_encoder', 'rope_bias_a', 'rope_bias_b', 'rope_bias_c')]
    todo, found = list(seeds), set()
    while todo:
        name = todo.pop()
        if name in found or not (ROOT/name).is_file():
            continue
        found.add(name)
        parts = name.removesuffix('.py').split('/')
        package = '.'.join(parts[:-1])
        for i in range(1, len(parts)):
            init = '/'.join(parts[:i])+'/__init__.py'
            if (ROOT/init).is_file() and init not in found:
                todo.append(init)
        for node in ast.walk(ast.parse((ROOT/name).read_text(encoding='utf-8-sig'))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = importlib.util.resolve_name('.'*node.level+(node.module or ''), package)
                else:
                    base = node.module or ''
                modules = [base]+[base+'.'+a.name for a in node.names]
            for module in modules:
                for candidate in (module.replace('.', '/')+'.py', module.replace('.', '/')+'/__init__.py'):
                    if (ROOT/candidate).is_file() and candidate not in found:
                        todo.append(candidate)
    return sorted(found | {'data/generation.py', 'data/tokenizor.json', 'configs/'+CONFIG.name})


def freeze():
    c = read(CONFIG)
    if OUT.exists() or DATA.exists() or (CONTRACT/'protocol.json').exists():
        raise FileExistsError('Use a new experiment ID')
    if sum(g['count'] for g in c['groups']) != 1024 or (c['topk'], c['steps']) != (4, 64):
        raise ValueError('Showcase specification changed')
    for kind in ('policy', 'encoder'):
        if sha(ROOT/c[kind]) != c[kind+'_sha256']:
            raise ValueError('Weight hash mismatch: '+kind)
    inputs = c['exclude_inputs']+c['origin_audits']
    code = dependency_files()
    engine = Path(importlib.util.find_spec('forsythia_hep').origin)
    p = dict(id=ID, config=c, code_sha256={n: sha(ROOT/n) for n in code},
             input_sha256={n: sha(ROOT/n) for n in inputs},
             engine_sha256={str(f): sha(f) for f in [engine, *engine.parent.glob('*.pyd')]},
             weight_sha256={c[k]: c[k+'_sha256'] for k in ('policy', 'encoder')},
             git_index_sha256=sha(ROOT/'.git/index'),
             dimension_monomial_max={str(d): max(monomial_capacities(d).values()) for d in range(2, 11)},
             primary_metric='source_mean_reduction_of_returned_shortest_expression', training_updates=0)
    OUT.mkdir(parents=True)
    for name in code:
        target = OUT/'source_backup'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    shutil.copy2(ROOT/'.git/index', OUT/'git_index.before')
    save(CONTRACT/'protocol.json', p)
    save(OUT/'frozen.json', dict(protocol_sha256=sha(CONTRACT/'protocol.json')))
    return dict(status='frozen', files=len(code), generation_caps=c['generation_caps'], evaluation_caps=c['evaluation_caps'])


def protocol():
    p = read(CONTRACT/'protocol.json')
    if sha(CONTRACT/'protocol.json') != read(OUT/'frozen.json')['protocol_sha256']:
        raise ValueError('Protocol changed')
    for name, expected in p['code_sha256'].items():
        if sha(OUT/'source_backup'/name) != expected:
            raise ValueError('Frozen source changed: '+name)
    for name, expected in {**p['input_sha256'], **p['engine_sha256'], **p['weight_sha256']}.items():
        if sha(ROOT/name) != expected:
            raise ValueError('Frozen dependency changed: '+name)
    return p


_last_storage_check = 0.


def resources(started, seconds):
    global _last_storage_check
    now = time.monotonic()
    settings = read(CONFIG)
    if now-started > seconds or shutil.disk_usage(ROOT).free < settings['free_bytes_floor']:
        raise RuntimeError('Resource limit reached')
    if now-_last_storage_check >= 5:
        _last_storage_check = now
        size = sum(f.stat().st_size for folder in (OUT, DATA) if folder.exists()
                   for f in folder.rglob('*') if f.is_file())
        if size > settings['artifact_bytes_cap']:
            raise RuntimeError('Artifact storage limit reached')


def generate():
    import numpy as np
    import forsythia_hep as api
    from project.reducibility.features import expression_key, validate
    from project.rl.point5_curriculum_data import original_config_generator, legacy_id
    p = protocol()
    c = p['config']
    if DATA.exists() or (OUT/'generation_started.json').exists():
        raise FileExistsError('Generation already started')
    DATA.mkdir()
    save(OUT/'generation_started.json', dict(caps=c['generation_caps']))
    ledger, rejected, rows, origins = Counts(c['generation_caps']), Counter(), [], []
    excluded_sources, excluded_inputs, excluded_origins = set(), set(), set()
    for name in c['exclude_inputs']:
        for r in read(ROOT/name):
            excluded_sources.add(r['source_id'])
            excluded_inputs.add(expression_key(r['tokens'], r['n_point']))
    for name in c['origin_audits']:
        with gzip.open(ROOT/name, 'rt', encoding='utf-8') as stream:
            for line in stream:
                r = json.loads(line)
                if r['accepted']:
                    excluded_origins.add(r['source_key'])
    rng = random.Random(c['generation_seed'])
    nrng = np.random.RandomState(c['generation_seed'])
    config_fn = original_config_generator(rng, nrng)
    started = time.monotonic()
    try:
        with gzip.open(OUT/'generation_attempts.jsonl.gz', 'xt', encoding='utf-8') as stream:
            for group in c['groups']:
                accepted = 0
                dims = [d for d in group['dimensions'] if max(monomial_capacities(d).values()) >= c['min_terms']]
                while accepted < group['count']:
                    resources(started, c['generation_seconds'])
                    ledger.hit('candidates')
                    helicities, dim = config_fn(4, rng.choice(dims))
                    helicities = list(map(int, helicities))
                    capacity = monomial_capacities(dim)[tuple(helicities)]
                    if capacity < c['min_terms']:
                        rejected['configuration_capacity'] += 1
                        continue
                    target_terms = rng.randint(c['min_terms'], min(c['max_terms'], capacity))
                    source = 0*api.abk(1, 2)
                    for _ in range(rng.randint(*c['source_terms'])):
                        ledger.hit('monomial_calls')
                        mono = api.random_monomial(helicities, dim)
                        validate(mono.tokenize(), 4, dim, helicities)
                        source += rng.choice((-3, -2, -1, 1, 2, 3))*mono
                    source = source.normal_order().expand()
                    original = source.tokenize()
                    sid, origin_key = legacy_id(original), expression_key(original, 4)
                    reason = None
                    if not 1 <= len(original) <= 5:
                        reason = 'source_length'
                    elif sid in excluded_sources or origin_key in excluded_origins or origin_key in excluded_inputs:
                        reason = 'duplicate_source'
                    current, calls = source, 0
                    if reason is None:
                        for _ in range(c['scramble_steps_per_candidate']):
                            ledger.hit('random_identity_calls')
                            calls += 1
                            current = (api.random_apply_schouten(current, 1) if rng.random() < .5
                                       else api.random_apply_momentum_conservation(current, 4, 1))[1]
                            tokens = current.tokenize()
                            validate(tokens, 4, dim, helicities)
                            if max(map(len, tokens), default=0) > 128 or len(tokens) > 128:
                                reason = 'process_capacity'
                                break
                            if target_terms <= len(tokens) <= c['max_terms']:
                                break
                    tokens = current.tokenize()
                    ikey = expression_key(tokens, 4)
                    if reason is None and not c['min_terms'] <= len(tokens) <= c['max_terms']:
                        reason = 'input_length'
                    if reason is None and (ikey in excluded_inputs or ikey in excluded_origins):
                        reason = 'duplicate_input'
                    if reason is None:
                        row = dict(index=len(rows), split='test', id=legacy_id(tokens), source_id=sid,
                                   n_point=4, dimension=dim, helicities_twice=helicities, terms=len(tokens),
                                   dimension_group=group['name'], tokens=tokens)
                        rows.append(row)
                        origins.append(dict(source_id=sid, source_key=origin_key, source_terms=len(original), tokens=original,
                                            input_id=row['id'], scramble_calls=calls, target_terms=target_terms))
                        excluded_sources.add(sid)
                        excluded_inputs.add(ikey)
                        excluded_origins.add(origin_key)
                        accepted += 1
                    else:
                        rejected[reason] += 1
                    stream.write(json.dumps(dict(candidate=ledger.values['candidates'], group=group['name'], dimension=dim,
                        accepted=reason is None, reason=reason, source_terms=len(original), terms=len(tokens), scramble_calls=calls))+'\n')
                    if reason is None and len(rows) % 32 == 0:
                        stream.flush()
                        print('DATA_PROGRESS '+json.dumps(dict(accepted=len(rows), group=group['name'], counts=ledger.values)), flush=True)
        assert len(rows) == len({r['source_id'] for r in rows}) == 1024
        save(DATA/'test.json', rows)
        save(DATA/'origins.json', origins)
        result = dict(status='completed', count=len(rows), counts=dict(ledger.values), rejections=dict(rejected),
            dimension_counts=dict(Counter(r['dimension'] for r in rows)), term_counts=dict(Counter(r['terms'] for r in rows)),
            mean_terms=st.fmean(r['terms'] for r in rows), mean_source_terms=st.fmean(r['source_terms'] for r in origins),
            native_rng_seeded=False, test_sha256=sha(DATA/'test.json'), origins_sha256=sha(DATA/'origins.json'),
            helicity_configurations=len({(r['dimension'], tuple(r['helicities_twice'])) for r in rows}),
            elapsed_seconds=time.monotonic()-started)
        save(DATA/'manifest.json', result)
        save(OUT/'generation_completed.json', result)
        return result
    except BaseException as exc:
        save(DATA/'partial_test.json', rows)
        save(OUT/'generation_failed.json', dict(error=repr(exc), counts=dict(ledger.values), accepted=len(rows)))
        raise


def evaluate():
    import torch
    import forsythia_hep as api
    from scripts.simplify_policy import PolicySimplifier
    from project.rl.packed32_common import expression_from_tokens, tensor_digest
    p = protocol()
    c = p['config']
    if (OUT/'evaluation_started.json').exists():
        raise FileExistsError('Evaluation already started')
    data_manifest = read(DATA/'manifest.json')
    if sha(DATA/'test.json') != data_manifest['test_sha256']:
        raise ValueError('Test inputs changed')
    rows = read(DATA/'test.json')
    assert len(rows) == 1024
    torch.set_num_threads(1)
    torch.manual_seed(c['inference_seed'])
    ledger, records = Counts(c['evaluation_caps']), []
    save(OUT/'evaluation_started.json', dict(caps=c['evaluation_caps'], test_sha256=sha(DATA/'test.json')))
    trace_folder = OUT/'traces'
    trace_folder.mkdir()
    started = time.monotonic()
    original_api = api.apply_identity_at
    runner = None
    handles = []
    try:
        ledger.hit('checkpoint_loads', 2)
        runner = PolicySimplifier(ROOT/c['policy'], ROOT/c['encoder'], device='cuda')
        initial_model = tensor_digest(runner.model.state_dict())
        handles.append(runner.model.encoder.register_forward_hook(lambda *args: ledger.hit('encoder_forwards')))
        value_calls = Counter()
        handles.append(runner.model.value_head.register_forward_hook(lambda *args: value_calls.update(['value'])))
        original_policy = runner.model.policy_logits

        def policy_logits(*args, **kwargs):
            ledger.hit('policy_forwards')
            return original_policy(*args, **kwargs)

        runner.model.policy_logits = policy_logits
        for row in rows:
            resources(started, c['evaluation_seconds'])
            ledger.hit('reconstructions')
            expression = expression_from_tokens(row['tokens'])
            trace_path = trace_folder/f'case_{row["index"]:04d}.jsonl.gz'
            with gzip.open(trace_path, 'xt', encoding='utf-8') as trace:
                serial = 0

                def apply_identity(parent, action, n_point):
                    nonlocal serial
                    ledger.hit('engine')
                    ledger.hit('sampling')
                    before = parent.tokenize()
                    children = original_api(parent, action, n_point)
                    if parent.tokenize() != before or n_point != 4:
                        raise RuntimeError('Engine parent/point mismatch')
                    serial += 1
                    trace.write(json.dumps(dict(step=serial, action=action, parent_tokens=before,
                                                child_tokens=children[0].tokenize(), n_point=n_point))+'\n')
                    return children

                api.apply_identity_at = apply_identity
                seed = int(content([ID, row['source_id'], c['inference_seed']])[:15], 16)
                best = runner.simplify(expression, 4, topk=4, budget_steps=64, seed=seed)
                api.apply_identity_at = original_api
                info = runner.last_info
                assert serial == info['steps'] <= 64
                assert info['best_terms'] == len(best.tokenize()) <= row['terms']
                record = dict(source_id=row['source_id'], index=row['index'], dimension=row['dimension'],
                    dimension_group=row['dimension_group'], seed=seed, **info, returned_tokens=best.tokenize(),
                    returned_expression=str(best), trace=trace_path.relative_to(ROOT).as_posix())
            record['trace_sha256'] = sha(trace_path)
            save(OUT/'records'/f'case_{row["index"]:04d}.json', record)
            records.append(record)
            if len(records) % 32 == 0:
                print('EVALUATION_PROGRESS '+json.dumps(dict(completed=len(records), **statistics(records), counts=dict(ledger.values))), flush=True)
        assert value_calls['value'] == 0
        assert tensor_digest(runner.model.state_dict()) == initial_model
        assert ledger.values['engine'] == ledger.values['encoder_forwards'] == ledger.values['policy_forwards']
        save(OUT/'records.json', records)
        result = dict(status='completed', counts=dict(ledger.values), model_tensors_unchanged=True, value_forwards=0,
                      training_updates=0, elapsed_seconds=time.monotonic()-started, results=statistics(records))
        save(OUT/'evaluation_completed.json', result)
        return result
    except BaseException as exc:
        save(OUT/'evaluation_failed.json', dict(error=repr(exc), counts=dict(ledger.values), completed=len(records)))
        raise
    finally:
        api.apply_identity_at = original_api
        for handle in handles:
            handle.remove()


def report():
    from project.reducibility.features import validate
    p = protocol()
    c = p['config']
    if (OUT/'final_audit.json').exists():
        raise FileExistsError('Report already completed')
    rows, origins, records = read(DATA/'test.json'), read(DATA/'origins.json'), read(OUT/'records.json')
    completed = read(OUT/'evaluation_completed.json')
    assert len(rows) == len(origins) == len(records) == 1024
    calls = 0
    for row, origin, record in zip(rows, origins, records):
        assert row['source_id'] == origin['source_id'] == record['source_id']
        assert row['terms'] == record['initial_terms'] and 21 <= row['terms'] <= 40
        assert len(origin['tokens']) == origin['source_terms'] <= 5
        validate(row['tokens'], 4, row['dimension'], row['helicities_twice'])
        validate(record['returned_tokens'], 4, row['dimension'], row['helicities_twice'])
        previous, best, best_tokens, count = row['tokens'], row['terms'], row['tokens'], 0
        assert sha(ROOT/record['trace']) == record['trace_sha256']
        with gzip.open(ROOT/record['trace'], 'rt', encoding='utf-8') as stream:
            for line, action_row in zip(stream, record['history'], strict=True):
                transition = json.loads(line)
                assert transition['parent_tokens'] == previous and transition['action'] == action_row['action']
                previous = transition['child_tokens']
                count += 1
                assert count == transition['step'] == action_row['step']
                assert len(previous) == action_row['after']
                if len(previous) < best:
                    best, best_tokens = len(previous), previous
                assert best == action_row['best_terms']
        assert count == record['steps'] <= 64 and len(previous) == record['final_terms']
        assert best_tokens == record['returned_tokens'] and best == record['best_terms']
        calls += count
    assert calls == completed['counts']['engine'] <= 65536
    results = {'all': statistics(records)}
    for g in c['groups']:
        subset = [r for r in records if group_name(r['dimension']) == g['name']]
        assert len(subset) == g['count']
        results[g['name']] = statistics(subset)
    audit = dict(passed=True, samples=1024, results=results, evaluation_counts=completed['counts'],
                 generation_counts=read(OUT/'generation_completed.json')['counts'], report_engine_calls=0,
                 report_model_calls=0, model_tensors_unchanged=completed['model_tensors_unchanged'])
    save(OUT/'final_audit.json', audit)
    save(CONTRACT/'results.json', audit)
    lines = ['# 四点振幅：1024条长表达式化简结果', '',
        '使用768维预训练encoder与PPO策略网络，对四点随机构型振幅进行逐步符号化简。每个输入在top4动作概率中采样一次，最多执行64步，返回路径上项数最少的表达式。', '',
        '| 量纲 | 输入数量 | 平均输入项数 | 平均返回项数 | 平均减项率 | 总项数减项率 | 平均执行步数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label, r in results.items():
        lines.append(f'| {label} | {r["count"]} | {r["mean_input_terms"]:.2f} | {r["mean_output_terms"]:.2f} | {r["mean_reduction"]:.2%} | {r["total_term_reduction"]:.2%} | {r["mean_steps"]:.2f} |')
    lines += ['', '平均减项率先计算每个输入的 `(输入项数－返回项数)/输入项数`，再对输入等权平均。总项数减项率按全部输入、返回项数的总和计算。', '',
        '## 数据与推理设置', '',
        '由1–5项、统一外线数/量纲/螺旋度的原始振幅，经Schouten恒等式与动量守恒随机扰乱，保留21–40项输入。量纲2–4、5–7、8–10三组分别342、341、341条；第一组实际为量纲4。数据与模型在评估前固定，每个输入执行一条top4采样路径。', '',
        f'模型：`{c["policy"]}`；encoder：`{c["encoder"]}`。完整数据分布见本地`datasets/{ID}/manifest.json`，每步路径见`artifacts/runs/{ID}/traces/`。', '',
        f'第64步或提前结束时的末状态平均减项率：{results["all"]["terminal_mean_reduction"]:.2%}；展示指标采用推理接口实际返回的最短式。', '',
        '## 实际计算量', '',
        f'数据生成：`{json.dumps(audit["generation_counts"], sort_keys=True)}`。推理：`{json.dumps(audit["evaluation_counts"], sort_keys=True)}`。训练更新0次；报告读取已保存路径完成核验。']
    with (CONTRACT/'summary.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines)+'\n')
    return audit


def run_phase(phase):
    if phase == 'plan':
        return read(CONFIG)
    return {'freeze': freeze, 'generate': generate, 'evaluate': evaluate, 'report': report}[phase]()
