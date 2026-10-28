"""Physical configurations, contrastive groups, and complete amplitudes."""
from __future__ import annotations
from typing import List,Tuple,Optional,Dict,Any,Callable,Iterator
import os
import numpy as np
import random
import itertools
import pickle
import gzip
from random import shuffle
import math
def random_monomial(*args,**kwargs):
    import forsythia_hep as api
    return api.random_monomial(*args,**kwargs)
def random_apply_schouten(*args,**kwargs):
    import forsythia_hep as api
    return api.random_apply_schouten(*args,**kwargs)
def random_apply_momentum_conservation(*args,**kwargs):
    import forsythia_hep as api
    return api.random_apply_momentum_conservation(*args,**kwargs)
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
def demo_amplitude_config(point, dimension, rng, nrng):
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
