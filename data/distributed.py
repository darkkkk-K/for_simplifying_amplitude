import os
import random
import pickle
import gzip
import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import torch.distributed as dist
from .tokenizer import tokenize


class AdvancedDistributedDataset(Dataset):
    """
    分布式数据集设计
    与DistributedSampler配合使用
    """

    def __init__(self, data_dir:str='./dataset', shuffle_files=True, max_files_in_memory=36):
        """
        不需要world_size和rank参数！
        DistributedSampler会处理数据分配
        """
        self.data_dir = data_dir
        self.shuffle_files = shuffle_files
        self.max_files_in_memory = max_files_in_memory

        # 加载所有文件
        self.file_list, self.metadata = self._discover_all_files()
        if shuffle_files:
            random.shuffle(self.file_list)

        # 预先计算总样本数（供DistributedSampler使用）
        self.total_samples = self.metadata["total_num_samples"]

        # 创建文件到样本范围的映射
        self.file_sample_ranges = self.metadata["filesample_index_mapping"]

        # 缓存管理
        self.loaded_files = {}
        self.load_order = []

    def _discover_all_files(self):
        """发现所有数据文件"""
        import glob
        pattern = os.path.join(self.data_dir, "amp_data_part_*.pkl.gz")
        files = sorted(glob.glob(pattern))
        metadata_path = os.path.join(self.data_dir, "metadata.pkl.gz")
        with gzip.open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        return files, metadata

    def __len__(self):
        """返回总样本数 - DistributedSampler需要这个"""
        return self.total_samples

    def __getitem__(self, idx):
        """
        根据全局索引获取样本
        DistributedSampler会调用这个方法，传入分配好的索引
        """
        # 1. 找到包含该索引的文件
        file_path, local_idx = self._find_file_for_index(idx)

        # 2. 确保文件已加载
        if file_path not in self.loaded_files:
            self._load_file_to_memory(file_path)

        # 3. 返回对应的样本
        return self.loaded_files[file_path][local_idx]

    def _find_file_for_index(self, global_idx):
        """根据全局索引找到对应的文件和局部索引"""
        for file_path, range_info in self.file_sample_ranges.items():
            if range_info['start_idx'] <= global_idx <= range_info['end_idx']:
                local_idx = global_idx - range_info['start_idx']
                return file_path, local_idx
        raise IndexError(f"索引 {global_idx} 超出范围")

    def _load_file_to_memory(self, file_path):
        """加载文件到内存（FIFO缓存）"""
        if file_path in self.loaded_files:
            return

        with gzip.open(file_path, 'rb') as f:
            file_data = pickle.load(f)

        self.loaded_files[file_path] = file_data
        self.load_order.append(file_path)

        # FIFO淘汰
        while len(self.loaded_files) > self.max_files_in_memory:
            oldest_file = self.load_order.pop(0)
            del self.loaded_files[oldest_file]