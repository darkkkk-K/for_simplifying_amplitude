"""Bounded experiments and provenance for the auxiliary classifier."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'reducibility_set_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def checked_id(run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,100}', run_id):
        raise ValueError('Invalid experiment ID')
    return run_id


def paths(run_id):
    checked_id(run_id)
    return ROOT / 'experiments' / run_id, ROOT / 'artifacts/runs' / run_id


def sources():
    files = list((ROOT / 'project/reducibility').glob('*.py'))
    files += [ROOT / p for p in ('scripts/generate_reducibility.py', 'scripts/train_reducibility.py',
              'configs/reducibility_v1.json', 'project/rl/legal_actions.py', 'data/tokenizor.json',
              'tests/test_reducibility.py')]
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in files}


def engine_info():
    spec = importlib.util.find_spec('forsythia_hep')
    if spec is None:
        raise RuntimeError('forsythia_hep is required')
    directory = Path(spec.origin).parent
    binaries = sorted(directory.glob('*.pyd')) + sorted(directory.glob('*.so'))
    return {'package': str(directory), 'binaries': {str(p): file_hash(p) for p in binaries},
            'python_seed_does_not_control_rust_rng': True}


def freeze(run_id, config, kind, extra=None):
    contract_dir, out = paths(run_id)
    if (contract_dir / 'protocol.json').exists() or out.exists():
        raise FileExistsError('Existing ID cannot be frozen/overwritten again')
    out.mkdir(parents=True)
    protocol = {'schema': SCHEMA, 'id': run_id, 'kind': kind, 'config': config,
                'sources': sources(), 'engine': engine_info(), 'extra': extra or {},
                'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'git_index_sha256': file_hash(ROOT / '.git/index'),
                'staged_diff_sha256': hashlib.sha256(subprocess.check_output(
                    ['git', 'diff', '--cached', '--binary'], cwd=ROOT)).hexdigest()}
    for rel in protocol['sources']:
        target = out / 'source_backup' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    save_json(contract_dir / 'protocol.json', protocol)
    return protocol


def check_protocol(run_id, kind):
    contract_dir, out = paths(run_id)
    protocol = json.loads((contract_dir / 'protocol.json').read_text(encoding='utf-8'))
    if protocol['kind'] != kind or protocol['sources'] != sources():
        raise RuntimeError('Frozen kind/source mismatch; do not edit running experiments')
    if protocol['engine'] != engine_info():
        raise RuntimeError('Engine differs from frozen build')
    if (out / 'started.json').exists():
        raise RuntimeError('This ID has already started; no implicit retries')
    save_json(out / 'started.json', {'time': time.time()})
    return protocol


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, limits=None):
        self.counts = {}
        self.limits = limits or {}

    def add(self, key, number=1):
        value = self.counts.get(key, 0) + number
        if key in self.limits and value > self.limits[key]:
            raise BudgetExceeded(key)
        self.counts[key] = value


def storage_bytes(directory):
    return sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file() and not p.is_symlink())
