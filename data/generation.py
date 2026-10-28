from forsythia_hep import *
from typing import List, Tuple, Optional, Dict, Any, Callable, Iterator
import os
import numpy as np
import random
import itertools
import pickle
import gzip
from random import shuffle
import math



'''
A = random_monomial([-1, 1, 2, -2], 10)
print(A)
print(A.tokenize())
B = random_apply_schouten(A, 2)
print(B[1].tokenize())
C = random_apply_momentum_conservation(A,4,1)
print(C[1].tokenize())
'''
'''
根据量纲和螺旋度构建散射振幅数据集。所有振幅外腿螺旋度之和应为整数(*2后为偶数)，振幅量纲应该大于等于所有外腿粒子螺旋度绝对值之和，
且如果sum 2h_i % 4 == 0，dim为偶数，若sum 2h_i % 4 !=0, 则dim为奇数，且螺旋度为负的粒子螺旋度之和要为整数
'''

def amplitude_config_gen(point: int,
                         dimension: int)->Tuple[List[int], int]:
    '''
    根据外腿数和维度随机生成振幅
    '''
    dim = dimension
    if dimension < 0:
        raise ValueError("Mass dimension must > 0!")

    if dimension % 2 == 0:
        hels_sum_double_abs = 4*random.randint(0, min(int(dimension/2), (5*point)//4))
    else:
        hels_sum_double_abs = 4*random.randint(0, min(int((dimension-1)/2), math.ceil((5*point-4)/4))) + 2

    hels_config_abs = np.random.multinomial(hels_sum_double_abs, [1/point]*point, size=1)[0]  # 按照多项式分布生成螺旋度绝对值，但是不保证某些大于5
    condition = hels_config_abs > 5
    illegal_idx = np.where(condition)[0]  # 大于5的违法2倍螺旋度索引
    illegal_hels_abs = hels_config_abs[illegal_idx]
    addend_alive = sum(illegal_hels_abs) - 5*len(illegal_idx)
    length =len(illegal_idx)
    random_plus_minus = 2*np.random.randint(0,2,size=len(hels_config_abs))-1
    if length == 0:
        hels_config = np.multiply(random_plus_minus, hels_config_abs)
    else:    # 使用最笨的办法，能保证O(n^3)以内的时间复杂度
        while addend_alive > 0:
            min_idx = np.argmin(hels_config_abs)
            hels_config_abs[min_idx] += 1
            addend_alive -= 1
        hels_config_abs[illegal_idx] = 5
    hels_config = np.multiply(random_plus_minus, hels_config_abs)
    condition_minus = hels_config < 0
    neg_hels_idx = np.where(condition_minus)[0]  # 负螺旋度粒子索引
    neg_hels = hels_config[neg_hels_idx]
    neg_hels_double_sum = -sum(neg_hels)
    if neg_hels_double_sum % 2 == 0:
        return hels_config.tolist(), dim
    else:
        condition_odd = neg_hels % 2 != 0
        local_odd_idx = np.where(condition_odd)[0]
        neg_odd_1st_global_idx = neg_hels_idx[local_odd_idx[0]]
        hels_config[neg_odd_1st_global_idx] = -hels_config[neg_odd_1st_global_idx]
        return hels_config.tolist(), dim

B = amplitude_config_gen(4, 6)

def randomly_scramble(amp_config: Tuple)->List[List[str]]:
    '''
    随机用schouten_identity或者动量守恒来扰乱表达式，由于表达式本身结构原因可能会扰乱失败，若扰乱失败则返回空列表。
    '''
    factor = np.random.randint(1, 11)
    amp = factor*random_monomial(amp_config[0], amp_config[1])
    amp_List = amp.tokenize()
    if len(amp_List) == 0:
        empty_sample = []
        return empty_sample
    amp_momscram = random_apply_momentum_conservation(amp, len(amp_config[0]), 1)[1]
    amp_schouscram = random_apply_schouten(amp, 1)[1]
    if amp_momscram == amp and amp_schouscram == amp:  # 表示不可扰乱
        empty_sample = []
        return empty_sample
    elif amp_momscram != amp and amp_schouscram == amp:
        rounds = random.randint(1,2)
        if rounds == 1:
            amp_scrambled = amp_momscram - amp
            amp_scrambled_list = amp_scrambled.tokenize()
        else:
            amp_scrambled = random_apply_momentum_conservation(amp_momscram, len(amp_config[0]), 1)[1] - amp
            amp_scrambled_list = amp_scrambled.tokenize()
    elif amp_momscram == amp and amp_schouscram != amp:
        rounds = random.randint(1,2)
        if rounds == 1:
            amp_scrambled = amp_schouscram - amp
            amp_scrambled_list = amp_scrambled.tokenize()
        else:
            amp_scrambled = random_apply_schouten(amp_schouscram, 1)[1] - amp
            amp_scrambled_list = amp_scrambled.tokenize()
    else:
        round_state = random.randint(1,6)
        if round_state == 1:
            amp_scrambled = amp_momscram - amp
            amp_scrambled_list = amp_scrambled.tokenize()
        elif round_state == 2:
            amp_scrambled = amp_schouscram - amp
            amp_scrambled_list = amp_scrambled.tokenize()
        elif round_state == 3:
            amp_scrambled = random_apply_momentum_conservation(amp_momscram, len(amp_config[0]), 1)[1] - amp
            amp_scrambled_list = amp_scrambled.tokenize()
        elif round_state == 4:
            amp_scrambled = random_apply_schouten(amp_schouscram, 1)[1] - amp
            amp_scrambled_list = amp_scrambled.tokenize()
        elif round_state == 5:
            amp_scrambled = random_apply_momentum_conservation(amp_schouscram, len(amp_config[0]), 1)[1] - amp
            amp_scrambled_list = amp_scrambled.tokenize()
        else:
            amp_scrambled = random_apply_schouten(amp_momscram, 1)[1] - amp
            amp_scrambled_list = amp_scrambled.tokenize()
    return amp_scrambled_list

'''
sample = randomly_scramble(B)
print(sample)
'''


def data_gen_for_pre(total_samples:int = 10000000,
    chunk_size: int = 100000, # 分块大小就设置为batch_size
    if_chunk: bool = False,
    output_file: str = "amp_data",
    output_dir: str = './dataset',
    compress: bool = True,
    temperature:float = 1.8,
    )->List[List[List[str]]]:
     point_candiadtes = [3, 4, 5, 6]
     point_weights = [np.exp(3.0/temperature), np.exp(4.0/temperature), np.exp(5.0/temperature), np.exp(6.0/temperature)]
     def valid_samples_generator():
        """生成有效样本的训练样本生成器"""
        generated_count = 0
        empty_count = 0
        while generated_count < total_samples:
            point = np.random.choice(point_candiadtes, p=np.array(point_weights/np.sum(point_weights)))
            dim = np.random.randint(2, (10*point)//2)
            sample_config = amplitude_config_gen(point, dim)
            sample = randomly_scramble(sample_config)
            if sample:  # 只保留非空样本
                yield sample    # 返回生成器对象，内存友好
                generated_count += 1

                # 进度显示
                if generated_count % 4096 == 0:
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

    # 如果指定了输出文件，保存到文件
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

def load_dataset(filename: str) -> Iterator[List[Any]]:
    """从文件加载数据集"""
    if filename.endswith('.gz'):
        with gzip.open(filename, 'rb') as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break
    else:
        with open(filename, 'rb') as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

'''
data_gen_for_pre(total_samples = 10000,
                 chunk_size = 50000,
                 if_chunk=False,
                 output_file = "amp_data.pkl.gz",
                 compress = False)
'''

def generate_distributed_dataset(total_samples:int=10000000,
         samples_per_file:int=500000,
         data_dir:str='./dataset',
         val_split:float=0.25,
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
        data_gen_for_pre(
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
        data_gen_for_pre(
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
# 测试生成数据集
t_i = time.time()
generate_distributed_dataset(total_samples=20000, samples_per_file=1000)
t_f = time.time()
delta_t = t_f-t_i
print(f"生成数据集用时：{delta_t:.2f}秒")
'''