import gzip
import pickle
import random
import threading
from collections import deque
from pathlib import Path
from random import shuffle
from typing import List, Tuple, Optional, Dict, Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from .tokenizer import tokenize
from forsythia_hep import *


'''
dataset生成abk form->List[List[str]]，每一项为random(0,99)<>^m[]^n，要保证螺旋度守恒，生成的形式为List[List[List[str]]]内层为一项，中层为一个表达式，外层为所有表达式集合
'''

class MyDataset(Dataset):
    '''
    自定义数据集形式，我们定义为List[List[List[str]]]
    '''
    def __init__(self, data:List[List[List[str]]]):
        super(MyDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)  # 样本数

    def __getitem__(self, idx):
        return self.data[idx]  # 一个样本

    def shuffle(self):
        shuffle(self.data)

    def pr(self):
        print(self.data)

def collate_fn(batch:List[List[List[str]]]):
    '''
    自定义分块，得到的一个batch为[T,seq_len]
    '''
    batch_tensor = tokenize(batch, input_batch=True)
    return batch_tensor

# === 分布式安全的样本采样 ===
# 采样策略：在遍历完当前 rank 负责的所有 shard 之前不放回；
# 遍历结束后重新打乱 shard 顺序再开启下一轮，避免不同 rank 之间重复读取。
_DATA_ROOT = Path(__file__).resolve().parent.parent / "datasets" / "rl"
_DEFAULT_SPLIT = "train"  # 可选 "train" / "val"
_SHARD_QUEUE: Optional[deque] = None
_CURRENT_SHARD: List[Any] = []
_CURRENT_IDX: int = 0
_SAMPLE_LOCK = threading.Lock()


def _rank_world() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _discover_shards(split: str) -> List[Path]:
    split_dir = _DATA_ROOT / f"{split}_data"
    pattern = "amp_data_part*.pkl*"
    shards = sorted(split_dir.glob(pattern))
    if not shards:
        raise FileNotFoundError(f"No dataset shards found under {split_dir}")
    rank, world_size = _rank_world()
    assigned = shards[rank::world_size]
    if not assigned:
        raise RuntimeError(
            f"Rank {rank} has no shards to read (world_size={world_size}, total_shards={len(shards)})"
        )
    return assigned


def _load_shard(path: Path) -> List[Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        data = pickle.load(f)
    # 数据在生成阶段已经打乱，这里保持原顺序即可；若需要进一步随机可在此 shuffle
    return data


def _reset_sampler(split: str) -> None:
    global _SHARD_QUEUE, _CURRENT_SHARD, _CURRENT_IDX
    shard_list = _discover_shards(split)
    random.shuffle(shard_list)  # 每轮重新打乱 shard 读取顺序，减少相关性
    _SHARD_QUEUE = deque(shard_list)
    _CURRENT_SHARD = []
    _CURRENT_IDX = 0


def sample_new_expressions(split: str = _DEFAULT_SPLIT) -> Tuple[SpExpr, int, int]:
    """
    从磁盘增量读取新的 (SpExpr, num_external_legs, num_sourced_terms) 样本。

    - 分布式：依据 rank 对 shard 切片，确保不同 rank 不会读取同一文件。
    - 采样：在耗尽当前 rank 的所有 shard 前不放回；全部用完后重新打乱 shard 顺序再循环。
    """

    global _CURRENT_IDX, _CURRENT_SHARD
    with _SAMPLE_LOCK:
        if _SHARD_QUEUE is None:
            _reset_sampler(split)

        # 如果当前 shard 用尽，切换到下一个；若所有 shard 用完则重置后再取。
        while _CURRENT_IDX >= len(_CURRENT_SHARD):
            if not _SHARD_QUEUE:
                _reset_sampler(split)
            shard_path = _SHARD_QUEUE.popleft()
            _CURRENT_SHARD = _load_shard(shard_path)
            _CURRENT_IDX = 0

        sample = _CURRENT_SHARD[_CURRENT_IDX]
        _CURRENT_IDX += 1
        return sample