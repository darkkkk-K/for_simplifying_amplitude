from forsythia_hep import *
from typing import List, Tuple, Optional, Dict, Any, Callable, Iterator
import os
from torch.utils.data import DataLoader, Dataset
from data.generation import amplitude_config_gen
import torch.distributed as dist
import numpy as np
import random
import itertools
import pickle
import gzip
from random import shuffle
import math
import time

def randomly_scramble(amp_config:Tuple, num_terms:int, max_scramble_times:int)->Tuple[SpExpr, int, int]:
    '''
    args:
    amp_config: Tuple, amplitude configuration from amplitude_config_gen, 包含粒子螺旋度构型和质量维度
    num_terms: int, number of monomials in the expression
    max_scramble_times: int, maximum number of scrambling attempts
    returns:
    sample: Tuple[SpExpr, int, int], 包含扰乱后的表达式，粒子数，初始项数
    说明：生成一个包含num_terms项的随机表达式，并对其进行扰乱，扰乱方式包括动量守恒和Schouten恒等式的应用，扰乱次数不超过max_scramble_times。
    如果扰乱后表达式的项数增加，则返回该表达式及其相关信息；否则返回空列表表示扰乱失败。
    '''
    factors = [random.randint(1, 10) for i in range(num_terms)]
    amp_list = [factors[i] * random_monomial(amp_config[0], amp_config[1]) for i in range(num_terms)]
    amp_src = sum(amp_list)
    lenth_initial = len(amp_src.terms())
    amp_momscram = random_apply_momentum_conservation(amp_src, len(amp_config[0]), 1)[1]
    amp_schouscram = random_apply_schouten(amp_src, 1)[1]
    if amp_momscram == amp_src and amp_schouscram == amp_src:  # 表示不可扰乱
        empty_sample = []
        return empty_sample
    for i in range(max_scramble_times):
        choice = random.choice([0, 1])
        if choice == 0:
            amp_src = random_apply_momentum_conservation(amp_src, len(amp_config[0]), 1)[1]
        else:
            amp_src = random_apply_schouten(amp_src, 1)[1]
    if len(amp_src.terms()) > lenth_initial:
        sample = (amp_src, len(amp_config[0]), lenth_initial)
        return sample
    else:
        empty_sample = []
        return empty_sample

'''
B = amplitude_config_gen(4, 6)
a = randomly_scramble(B, 3, 128)
print(a)
with open('a.pkl', 'wb') as f:
    pickle.dump([a], f)
'''


def data_gen_for_RL(total_samples:int = 524288,
    chunk_size: int = 100000, # 分块大小就设置为batch_size
    if_chunk: bool = False,
    output_file: str = "amp_data",
    output_dir: str = './RLdataset',
    max_scramble_times: int = 128,
    compress: bool = True,
    temperature:float = 1.8,
    )->List[Tuple[SpExpr, int, int]]:
    '''
    生成用于RL训练的数据集，形式为List[Tuple[SpExpr, int, int]]，列表中每一项为(表达式，粒子数，初始项数)
    '''
    point_candiadtes = [3, 4, 5, 6]
    point_weights = [np.exp(3.0/temperature), np.exp(4.0/temperature), np.exp(5.0/temperature), np.exp(6.0/temperature)]

    def valid_samples_generator():
        """生成有效样本的训练样本生成器"""
        generated_count = 0
        empty_count = 0
        while generated_count < total_samples:
            scram_times = random.randint(3, max_scramble_times)
            num_terms = random.randint(1, 5)
            point = np.random.choice(point_candiadtes, p=np.array(point_weights/np.sum(point_weights)))
            dim = np.random.randint(2, (10*point)//2)
            sample_config = amplitude_config_gen(point, dim)
            sample = randomly_scramble(amp_config=sample_config, num_terms=num_terms, max_scramble_times=scram_times)
            if sample:  # 只保留非空样本
                yield sample    # 返回生成器对象，内存友好
                generated_count += 1
                
                # 进度显示
                if generated_count % 12800 == 0:
                    print(f"已生成 {generated_count}/{total_samples} 个有效样本，"
                          f"跳过 {empty_count} 个空样本")
            else:
                empty_count += 1
    def chunked_generator():
        """将样本分块返回，返回分块的生成器，生成器中每个对象是一个List[List[List[str]]]，也可不分块"""
        iterator = valid_samples_generator()
        while if_chunk:
            chunk = list(itertools.islice(iterator, chunk_size))  # 用islice时会消耗迭代器里的元素，因此当分不出块的时候就会break
            if not chunk:
                break
            yield chunk
        if not if_chunk:
            amp_data = list(iterator)
            yield amp_data

    if output_file:
        _save_to_file(chunked_generator(), filename=output_file, output_dir=output_dir, compress=True)
        print(f"数据集已保存到: {output_file}")
        return None
    else:
        # 返回生成器用于流式处理
        return chunked_generator()
    
    
def _save_to_file(data_generator: Iterator, filename: str, output_dir:str="./dataset",compress: bool = True):
    """将数据保存到文件"""
    os.makedirs(output_dir, exist_ok=True)

    if compress:
        filename = filename if filename.endswith('.gz') else filename + '.gz'
        filepath = os.path.join(output_dir, filename)
        with gzip.open(filepath, 'wb') as f:  # 'wb'模式会自动创建文件
            for chunk in data_generator:
                pickle.dump(chunk, f)    
    else:
        with open(filepath, 'wb') as f:
            for chunk in data_generator:
                pickle.dump(chunk, f)

def generate_distributed_dataset(total_samples:int=524288,
         samples_per_file:int=2560,
         data_dir:str='./RLdataset',
         val_split:float=0.20,
         temperature:float=1.8,
         worker_id:int=0,
         random_seed:int=42
         ):  
    '''
    分布式数据生成器，将数据分为800个文件，每个文件12800个样本。同时会生成元数据文件以记录样本总量以及单文件文件样本-全局索引映射。work_id支持多进程处理
    '''
    random.seed(random_seed+worker_id)
    np.random.seed(random_seed+worker_id)
    num_train_samples = math.ceil((1-val_split)*total_samples)
    num_val_samples = total_samples - num_train_samples
    num_train_files = (num_train_samples + samples_per_file - 1) // samples_per_file 
    num_val_files = (num_val_samples + samples_per_file - 1) // samples_per_file
    train_dir = os.path.join(data_dir, "train_data")
    val_dir = os.path.join(data_dir, "val_data")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    print(f"开始生成数据集并划分训练集/验证集...")
    print(f"总样本数: {total_samples:,}")
    print(f"训练样本数: {num_train_samples:,}")
    print(f"每个文件样本数: {samples_per_file:,}")
    print(f"文件数量: {num_train_files + num_val_files}")
    print(f"输出目录: {data_dir}")

    metadata_train = {}
    metadata_val = {}
    metadata_train["total_num_samples"] = num_train_samples 
    metadata_train["num_files"] = num_train_files
    metadata_train["filesample_index_mapping"] = {}
    metadata_val["total_num_samples"] = num_val_samples
    metadata_val["num_files"] = num_val_files
    metadata_val["filesample_index_mapping"] = {}
    current_idx_train = 0 + worker_id*num_train_samples
    current_idx_val = 0 + worker_id*num_val_samples
    for i in range(num_train_files):
        # 计算当前文件应包含的样本数（最后一个文件可能较少）
        current_file_samples = min(samples_per_file, num_train_samples - i * samples_per_file)

        filename = f"amp_data_part{i+num_train_files*worker_id:03d}.pkl.gz"
        file_path = os.path.join(train_dir, filename)
        print(f"生成文件{i+worker_id+1}/{num_train_files}: {filename}")
        data_gen_for_RL(
            total_samples=current_file_samples,
            chunk_size=10000,
            if_chunk=False,
            output_file=filename,
            output_dir=train_dir,
            compress=True,
            temperature=temperature)
        metadata_train["filesample_index_mapping"][file_path]={
                 'start_idx': current_idx_train,
                 'end_idx': current_idx_train+current_file_samples-1,
                 'num_samples': current_file_samples}
        current_idx_train += current_file_samples
    '''
    metadata_train_name = f"metadata{worker_id:02d}.pkl.gz"
    metadata_train_path = os.path.join(train_dir, metadata_train_name)
    with gzip.open(metadata_train_path, 'wb') as f:
        pickle.dump(metadata_train, f)
    '''
    print(f"训练集生成完成！文件保存在: {train_dir}")
    for i in range(num_val_files):
        # 计算当前文件应包含的样本数（最后一个文件可能较少）
        current_file_samples = min(samples_per_file, num_val_samples - i * samples_per_file)

        filename = f"amp_data_part{i+num_val_files*worker_id:03d}.pkl.gz"
        file_path = os.path.join(val_dir, filename)
        print(f"生成文件{i+worker_id+1}/{num_val_files}: {filename}")
        data_gen_for_RL(
            total_samples=current_file_samples,
            chunk_size=10000,
            if_chunk=False,
            output_file=filename,
            output_dir=val_dir,
            compress=True,
            temperature=temperature)
        metadata_val["filesample_index_mapping"][file_path]={
                 'start_idx': current_idx_val,
                 'end_idx': current_idx_val+current_file_samples-1,
                 'num_samples': current_file_samples}
        current_idx_val += current_file_samples
    '''
    metadata_val_name = f"metadata{worker_id:02d}.pkl.gz"
    metadata_val_path = os.path.join(val_dir, metadata_val_name)
    with gzip.open(metadata_val_path, 'wb') as f:
        pickle.dump(metadata_val, f)
    '''
    print(f"验证集生成完成！文件保存在: {val_dir}")
    return metadata_train, metadata_val

'''
data_gen_for_RL(total_samples = 2560,
                 chunk_size = 50000,
                 if_chunk=False,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                 output_file = "testRLdata.pkl.gz",
                 output_dir = './RLdataset', 
                 compress = True)
'''