"""JSONL sharding; every input row is written exactly once."""
import gzip
import json


def write_shards(directory, split, records, chunk_size=1024):
    if chunk_size <= 0:
        raise ValueError('Positive chunk size required')
    for start in range(0, len(records), chunk_size):
        path = directory / f'{split}_{start//chunk_size:03d}.jsonl.gz'
        if path.exists():
            raise FileExistsError(path)
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            for record in records[start:start + chunk_size]:
                f.write(json.dumps(record, ensure_ascii=True) + '\n')
