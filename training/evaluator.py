import torch
from typing import List, Literal, Optional
from typing import Dict, List, Tuple, Any
import torch.nn.functional as F
from data.tokenizer import tokenize
from data.generation import data_gen_for_pre
from models.encoder import ClusteringMachine1
from training.loss import NTXentloss
'''
用于评估第一层聚类器的聚类效果
'''

def comprehensive_evaluate(src:torch.Tensor,
                output:torch.Tensor,
                device:torch.device):  # 输入是[B,T,seq_len](神经网络输入)以及[B,T,D](神经网络输出)张量，已完成掩码
    '''
    计算：
        1.一个标签内部每个向量与标签内归一化平均向量的差向量的模方之和/样本中的向量数量，越小越好；
        2.不同标签间平均向量构型的余弦互能，
    '''
    B,T,D = output.shape
    pad = (src != 0).float().sum(dim=-1)
    mask = (pad != 0).float()  # [B, T]的掩码，填充位置为0
    num_term = mask.sum(dim=-1) # [B]，代表每个样本有多少项
    num_pad_term = torch.tensor([T]*B).float().to(device)-num_term # 代表每个样本有多少掩码项
    output_sum = output.sum(dim=1) # [B, D]
    output_avg = F.normalize(output_sum, dim=-1) # [B, D]
    diff_from_avg= output - output_avg.unsqueeze(1)  # [B,T,D]每一个向量的与样本平均的差值向量
    diff_len = (torch.einsum('ijk,ijk->ij', diff_from_avg, diff_from_avg).sum(dim=-1) - num_pad_term) / num_term # [B]差值向量模方和/项数，减去了掩码项
    diff_avg = (diff_len.sum(dim=0) / B).item()  # 越接近0表示组内凝聚度越好
    avg_sim = torch.einsum('ik,jk->ij', output_avg, output_avg).sum(dim=-1) - torch.ones(B).float().to(device)  # 不同组平均向量之间的余弦相似度矩阵[B]，每一行为该点对其他的点的“相互作用能”
    energy = (avg_sim/2).sum(dim=0).item()  # 得到总能量，要求越接近0越好

    return diff_avg, energy
'''
device = torch.device("cuda")
cret = NTXentloss(temperature= 0.1)
test_model = ClusteringMachine1(
        vocab_size = 46,
        emb_dim = 16,
        numhead = 8,
        numenc1 = 3,
        fnn_dim = 64,
        dropout = 0.1
    ).to(device)
test_model.eval()
src = data_gen(3,6)
src_in = tokenize(src, input_batch=True, device=device)
print(src_in)
output = test_model(src_in)
print(output)
loss = cret(output)
diff, energy = comprehensive_evaluate(src = src_in, output=output, device=device)
print(diff)
print(energy)
print(loss)
'''
