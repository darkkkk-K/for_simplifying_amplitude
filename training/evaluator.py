import torch
import torch.nn.functional as F
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
