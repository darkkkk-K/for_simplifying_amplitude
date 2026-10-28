import json
import os
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any, Callable
import time

'''
分词器
'''

_tokenizor_json_path = os.path.join(os.path.dirname(__file__), 'tokenizor.json')
with open(_tokenizor_json_path, 'r') as f:
    tokenizor = json.load(f)

# A = ['0','5','<12>','[12]','_','6']
# C = [[['0','5','<12>','[12]','_','6'],['[12]','_','6'],['5','<12>','[12]']],[['<12>','[12]'],['0','5','<12>']]]
# B = list(map(tokenizor.get, A))
# print(B)
# print(C)

def variable_chunking(data:List[Any], chunk_lengths:List[int]):

    chunks = []
    start = 0

    for i, length in enumerate(chunk_lengths):
        end = start + length
        chunks.append(data[start:end])
        start = end
    return chunks

def tokenize(batch_expressions: List[List[List[str]]],
             input_batch: True,
             )->torch.Tensor|List[torch.Tensor]:     # 内层列表形如['0','5','<12>','[12]','_','6']，中层列表为一个表达式/一个正样本
    '''
    这是分词器，若输入的是一个batch，则将一个形如List[List[List[str]]]的原始振幅样本列表转换为形状为[B,T,seq_len]的torch.Tensor，若输入的是整个数据集，
    则只返回List[torch.Tensor],即每一个样本的张量[T,seq_len]组成的列表。映射详情见tokenizor.json。
    '''
    B = len(batch_expressions)
    T = [len(batch_expressions[i]) for i in range(B)]  # 返回每个样本的项数
    batch_tokentensor = [[torch.tensor(list(map(tokenizor.get, batch_expressions[i][j]))) for j in range(len(batch_expressions[i]))] for i in range(len(batch_expressions))]   # List[List[torch.tensor]]
    if input_batch == True:
        term_List = [item for sublist in batch_tokentensor for item in sublist]  # 得到所有项张量组成的列表List[torch.Tensor]
        term_tensors = nn.utils.rnn.pad_sequence(term_List, batch_first=True, padding_value=0)   # 得到形如(B*T,seq_l)的张量，内层seq_len掩码，接下来进行解包
        batch_List = variable_chunking(term_tensors, T)  # List[tensor]，每个tensor为[T_i, seq_len]，List传给collate_fn处理
        batch_tokens_per_sample = nn.utils.rnn.pad_sequence(batch_List, batch_first=True, padding_value=0)   # 外层T掩码，得到掩码好的batch张量
        return batch_tokens_per_sample
    else:
        batch_list_per_expression = [nn.utils.rnn.pad_sequence(batch_tokentensor[i]) for i in range(B)] # 内层为[T_i,seq_i]
        return batch_list_per_expression
# D = tokenize(C, input_batch=True)
# print(D)

def collate_fn(batch:List[List[List[str]]]):
    '''
    自定义分块，得到的一个batch为[T,seq_len]
    '''
    batch_tensor = tokenize(batch, input_batch=True)
    return batch_tensor