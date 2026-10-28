import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Callable
import numpy as np
import math
import json
import os
from data.vocabulary import reverse_tokenize
from .rope_bias_c import FlexibleConstrainedLearnableRoPE
from .custom_encoder import BatchFirstTransformerEncoderLayerWithCustomAttention, BatchFirstTransformerEncoder

tokenizor = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10}

reverse_tokenizor = reverse_tokenize(tokenizor)
token_id_of_bracket = [key for key, value in reverse_tokenizor.items()]

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim = 512, dropout = 0.1, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2)*(-math.log(10000.0)/emb_dim))  # aiayn原文经典位置编码的分母

        pe = torch.zeros(max_len, emb_dim)
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)  # 暂时用三角函数位置编码，后续用更高级的

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # x:[B*T, seq_len, emb_dim]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x + self.pe[:, :x.size(1), :].to(device) # 只动态取到seq_len而非max_len, 动态填充,
        return self.dropout(x)

class ClusteringMachine1(nn.Module):  # 预训练的模型
    '''模型初始化'''
    def __init__(self, vocab_size, emb_dim = 512, numhead = 8, numenc1 = 2, fnn_dim = 2048, dropout = 0.1):
        super(ClusteringMachine1, self).__init__()
        self.emb_dim = emb_dim
        self.embdding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = emb_dim
        )
        self.pos_encoding = PositionalEncoding(
            emb_dim = emb_dim,
            dropout = dropout
        )
        enc_layer = TransformerEncoderLayer(
            d_model = emb_dim,
            nhead = numhead,
            dim_feedforward = fnn_dim,
            batch_first = True,     ## 训练输入的形状为[batch_size, seq_len](传统上是相反的，但是这样更符合思维习惯)
            dropout = dropout,
            activation = 'relu'
        )
        self.transformer = TransformerEncoder(
            enc_layer,
            num_layers = numenc1
        )
        self.projector = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.last_pos_bias = nn.Parameter(torch.tensor(2.0))  # 注意力偏置，参数可学习，初始设置为2.0


    def forward(self, src:torch.Tensor, RL:bool = False):     ## src为[batch_size, set_size(一个批次中的项数), seq_len(一个项的序列token长度)]， 推理阶段输入的是[1, set_size, seq_len]
        '''
        需要得到除了合成的项向量之外，还需要得到离合成向量最近的头两个spinor bracket。
        '''
        batch_size, set_size, seq_len = src.shape
        src_re = src.reshape(batch_size*set_size, seq_len)  # 将输入形状转换为[batch_size*set_size, seq_len]
        padding_mask = (src_re == 0)   # 设置padding_value = 0
        src_emb = self.pos_encoding(self.embdding(src_re))

        output_emb = self.transformer(
            src = src_emb,
            src_key_padding_mask = padding_mask
        )
        output = self.projector(output_emb)   # 先输出为[batch_size*set_size, seq_len, emb_dim]

        mask = (src_re != 0).float()
        out = (output*mask.unsqueeze(-1)).sum(dim=1)  # [B*T,D]对一个项中所有token的嵌入向量求和，然后下一步直接归一化实现平均，掩码位置的向量一定为0向量
        output_norm = F.normalize(out, dim = -1)
        final_out = output_norm.reshape(batch_size,set_size,self.emb_dim)                      # 对output_norm第一维做展平处理
        '''
        logits_1 = (output*mask.unsqueeze(-1)).reshape(batch_size,set_size,seq_len,self.emb_dim) # [B,T,D]
        vec_per_token = F.normalize(logits_1, dim=-1)   # [B,T,D] vectors of tokens
        sim_to_termvec = torch.einsum('ijkl,ijl->ijk', vec_per_token, final_out)  # 每一项中每个token对项向量的相似度
        spinor_bracket_tensor = torch.isin(src.int(), torch.tensor(token_id_of_bracket).int()) # spinor bracket的位置上为1[B,T,s]
        sim_to_termvec_eff = sim_to_termvec*spinor_bracket_tensor.float()  # spinor bracket位置上的相似度才保存[B,T,s]
        src_in_pad = (src != 0).float() # [B,T,s]填充位置全为0，可用unsqueeze(-1)增加最后一维可和其他相关量广播计算
        sim_to_termvec_masked = sim_to_termvec.clone()
        sim_to_termvec_masked[~spinor_bracket_tensor] = -100 # 克隆一个相似度张量副本，将非有效位置的相似度张量元素设置为-100，远小于相似度下限-1
        top2_sim, relative_idx = torch.topk(sim_to_termvec_masked, k=2, dim=-1, largest=True, sorted=True)  # 返回top2的索引(根据相似度sorted)以及相似度值[B,T,2]]
        valid_mask= (top2_sim > -2)  #合法位置[B,T,2]，有可能找出来小于两个bracket
        cumsum_spinor = spinor_bracket_tensor.long().cumsum(dim=-1) - 1  # 对应spinorbracket位置值为该项第几个bracket
        position_map = torch.where(spinor_bracket_tensor, cumsum_spinor, torch.tensor(-16, dtype=torch.long, device=src.device)) # 无效位置设置为-16[B,T,s]
        true_position = torch.gather(position_map, dim = -1, index = relative_idx) # [B, T, 2]找到top2相似度在原项中是第几个bracket(按正则排序)
        true_position[~valid_mask] = -16
        '''
        '''
        if RL:
            Simplify_search_info = List[List[List[int]]]  # 用来储存化简从哪个因子开始
            search_position = self._find_priority_search(src=src, mask=mask, tgt=output, final_tgt=final_out)
            Simplify_search_info = [[search_position[i,j,:].tolist() for j in range(set_size)] for i in range(batch_size)]
            return final_out, Simplify_search_info  # 如果设置为强化学习模式还要带优先搜索的信息
        '''
        if RL:
            out = (output*(mask.unsqueeze(-1))) # [batch_size*set_size, seq_len, emb_dim]
            return out
        return final_out    # 输出维度为[batch_size, set_size, emb_dim]，为每个term的嵌入向量，就用各个token相加后归一化来表示，输出已经归一化，方便后续处理

    def _find_priority_search(self, src:torch.tensor, mask:torch.tensor, tgt:torch.tensor, final_tgt:torch.tensor):
        batch_size, set_size, seq_len = src.shape
        device = src.device
        logits_1 = (tgt*mask.unsqueeze(-1)).reshape(batch_size,set_size,seq_len,self.emb_dim) # [B,T,D]
        vec_per_token = F.normalize(logits_1, dim=-1)   # [B,T,D] vectors of tokens
        sim_to_termvec = torch.einsum('ijkl,ijl->ijk', vec_per_token, final_tgt)  # 每一项中每个token对项向量的相似度
        spinor_bracket_tensor = torch.isin(src.int(), torch.tensor(token_id_of_bracket).int().to(device)) # spinor bracket的位置上为1[B,T,s]
        sim_to_termvec_eff = sim_to_termvec*spinor_bracket_tensor.float()  # spinor bracket位置上的相似度才保存[B,T,s]
        src_in_pad = (src != 0).float() # [B,T,s]填充位置全为0，可用unsqueeze(-1)增加最后一维可和其他相关量广播计算
        sim_to_termvec_masked = sim_to_termvec.clone()
        sim_to_termvec_masked[~spinor_bracket_tensor] = -100 # 克隆一个相似度张量副本，将非有效位置的相似度张量元素设置为-100，远小于相似度下限-1
        top2_sim, relative_idx = torch.topk(sim_to_termvec_masked, k=2, dim=-1, largest=True, sorted=True)  # 返回top2的索引(根据相似度sorted)以及相似度值[B,T,2]]
        valid_mask= (top2_sim > -2)  #合法位置[B,T,2]，有可能找出来小于两个bracket
        cumsum_spinor = spinor_bracket_tensor.long().cumsum(dim=-1) - 1  # 对应spinorbracket位置值为该项第几个bracket
        position_map = torch.where(spinor_bracket_tensor, cumsum_spinor, torch.tensor(-16, dtype=torch.long, device=src.device)) # 无效位置设置为-16[B,T,s]
        true_position = torch.gather(position_map, dim = -1, index = relative_idx) # [B, T, 2]找到top2相似度在原项中是第几个bracket(按正则排序)
        true_position[~valid_mask] = -16

        return true_position

class ClusteringMachine1_RoPE(nn.Module):
    '''该class需要自定义QKV注意力矩阵，加入了RoPE位置编码和最后两个位置的注意力偏置'''
    def __init__(self, vocab_size, emb_dim = 768, numhead = 8, numenc1 = 4, fnn_dim = 3072, dropout = 0.1, RoPE_mode:str='shared_learnable'):
        super(ClusteringMachine1_RoPE, self).__init__()
        self.emb_dim = emb_dim
        self.embdding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = emb_dim
        )
        self.transformer = BatchFirstTransformerEncoder(
            encoder_layer=BatchFirstTransformerEncoderLayerWithCustomAttention,
            d_model = emb_dim,
            fnn_dim=fnn_dim,
            num_head = numhead,
            num_layers = numenc1,
            RoPE_mode=RoPE_mode
        )
        self.projector = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
    def forward(self, src:torch.Tensor, RL:bool = False):     ## src为[batch_size, set_size(一个批次中的项数), seq_len(一个项的序列token长度)]， 推理阶段输入的是[1, set_size, seq_len]
        '''
        需要得到除了合成的项向量之外，还需要得到离合成向量最近的头两个spinor bracket。
        '''
        batch_size, set_size, seq_len = src.shape
        src_re = src.reshape(batch_size*set_size, seq_len)  # 将输入形状转换为[batch_size*set_size, seq_len]
        #print(src_re)
        padding_mask = (src_re == 0)   # 设置padding_value = 0
        src_emb = self.embdding(src_re)
        #print(src_emb)
        output_emb = self.transformer(
            src = src_emb,
            src_key_padding_mask = padding_mask
        )
        output = self.projector(output_emb)   # 先输出为[batch_size*set_size, seq_len, emb_dim]
        mask = (src_re != 0).float()
        if RL:
            output_for_RL = output*(mask.unsqueeze(-1))
            return output_for_RL
        out = (output*(mask.unsqueeze(-1))).sum(dim=1)  # [B*T,D]对一个项中所有token的嵌入向量求和，然后下一步直接归一化实现平均，掩码位置的向量一定为0向量
        output_norm = F.normalize(out, dim = -1)
        final_out = output_norm.reshape(batch_size,set_size,self.emb_dim)                      # 对output_norm第一维做展平处理


        return final_out    # 输出维度为[batch_size, set_size, emb_dim]，为每个term的嵌入向量，就用各个token相加后归一化来表示，输出已经归一化，方便后续处理

    def _find_priority_search(self, src:torch.tensor, mask:torch.tensor, tgt:torch.tensor, final_tgt:torch.tensor):
        batch_size, set_size, seq_len = src.shape
        device = src.device
        logits_1 = (tgt*mask.unsqueeze(-1)).reshape(batch_size,set_size,seq_len,self.emb_dim) # [B,T, s ,D]
        vec_per_token = F.normalize(logits_1, dim=-1)   # [B,T,D] vectors of tokens
        sim_to_termvec = torch.einsum('ijkl,ijl->ijk', vec_per_token, final_tgt)  # 每一项中每个token对项向量的相似度
        spinor_bracket_tensor = torch.isin(src.int(), torch.tensor(token_id_of_bracket).int().to(device)) # spinor bracket的位置上为1[B,T,s]
        sim_to_termvec_eff = sim_to_termvec*spinor_bracket_tensor.float()  # spinor bracket位置上的相似度才保存[B,T,s]
        src_in_pad = (src != 0).float() # [B,T,s]填充位置全为0，可用unsqueeze(-1)增加最后一维可和其他相关量广播计算
        sim_to_termvec_masked = sim_to_termvec.clone()
        sim_to_termvec_masked[~spinor_bracket_tensor] = -100 # 克隆一个相似度张量副本，将非有效位置的相似度张量元素设置为-100，远小于相似度下限-1
        top2_sim, relative_idx = torch.topk(sim_to_termvec_masked, k=2, dim=-1, largest=True, sorted=True)  # 返回top2的索引(根据相似度sorted)以及相似度值[B,T,2]]
        valid_mask= (top2_sim > -2)  #合法位置[B,T,2]，有可能找出来小于两个bracket
        cumsum_spinor = spinor_bracket_tensor.long().cumsum(dim=-1) - 1  # 对应spinorbracket位置值为该项第几个bracket
        position_map = torch.where(spinor_bracket_tensor, cumsum_spinor, torch.tensor(-16, dtype=torch.long, device=src.device)) # 无效位置设置为-16[B,T,s]
        true_position = torch.gather(position_map, dim = -1, index = relative_idx) # [B, T, 2]找到top2相似度在原项中是第几个bracket(按正则排序)
        true_position[~valid_mask] = -16

        return true_position




class ClusteringMachine2(nn.Module):  # 这个网络的目的为在实际的化简任务中，给前面输出后的每一个向量(代表散射振幅的一项)加入注意力耦合，让后续算法实现分组
    def __init__(self, vocab_size, emb_dim = 512, numhead = 8, numenc2 = 4, fnn_dim = 2048, dropout = 0.1):
        super(ClusteringMachine2, self).__init__()
        self.emb_dim = emb_dim
        layer1 = nn.TransformerEncoderLayer(
            d_model = emb_dim,
            nhead = numhead,
            dim_feedforward = fnn_dim,
            batch_first = True,     ## 这一层在强化学习训练输入的形状为[batch_size, set_size, emb_dim], 填充在上一步输出已经做了
            dropout = dropout,
            activation = 'relu'
        )
        self.transformer = TransformerEncoder(
            layer1,
            num_layers = numenc2
        )
        self.projector2 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
    def forward(self, src, src_in):  # src_in为最初输入[batch_size, set_size, seq_len], 这里需要这个输入是要创造逐项的掩码
        src_pad = (src_in != 0).float()
        src_in_pad = (src_in != 0).float().sum(dim=-1)  # [B, T]，填充项的位置自然为0
        padding_mask = (src_in_pad == 0)
        RLout = self.transformer(
            src = src,
            src_key_padding_mask = padding_mask
        )
        mask = (src_in_pad != 0).float()  # [B, T]下一步乘法自动广播
        out = RLout*mask.unsqueeze(-1)
        RLout_norm = F.normalize(out, dim=-1)   # 得到了归一化的张量[batch_size,set_size,emb_dim]，同样输出的在填充项上的向量一定是0向量
        return RLout_norm, src_pad


_config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'pretrain.json')
with open(_config_path, 'r') as f:   # 'r'表示只可读
    configs = json.load(f)