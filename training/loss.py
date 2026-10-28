import torch
import torch.nn.functional as F
import math

'''
本文件为预训练损失函数
'''

def remove_diagonal(tensor)-> torch.Tensor:
    """
    移除张量的对角线元素并调整为 [n-1, n-1] 的形状

    参数:
    tensor (torch.Tensor): 形状为 [n, n] 的输入张量

    返回:
    torch.Tensor: 形状为 [n-1, n-1] 的输出张量
    """
    n = tensor.size(0)
    # 创建一个掩码，对角线位置为 False，其他位置为 True
    mask = ~torch.eye(n, dtype=torch.bool, device=tensor.device)  # ~表示取反，非对角元为真
    # 应用掩码获取非对角线元素，按行遍历得到一维张量
    non_diag_elements = tensor[mask]
    # 将非对角线元素重新塑形为 [n-1, n-1]
    return non_diag_elements.reshape(n, n-1)



class NTXentloss(torch.nn.Module):
    '''
    改良版交叉熵函数，作为预训练阶段的损失函数。
    '''
    def __init__(self, temperature = 0.1):
        super(NTXentloss, self).__init__()
        self.temperature = temperature
        self.cossim = torch.nn.CosineSimilarity(dim = -1)

    def forward(self, embeddings):
        # embeddings为一个包含正样本的batch，形状的[batch_size, num_term, emb_dim], 掩码在上一步已经完成，batch数量就为标签数量，num_terms为一个标签中的样本数量/项数
        B, T, D = embeddings.shape
        device = embeddings.device
        embeddings_len = torch.einsum('ijk,ijk->ij', embeddings, embeddings) # 返回[B, T]为每组样本中每个向量的模方
        mask = (embeddings_len != 0)   # 样本掩码[B, T]
        num_of_term_perlabel = (mask.int()).sum(dim=-1)   # 返回[B]，为每组正样本的实际项数
        num_total_term = num_of_term_perlabel.sum(dim=-1).item()

        mask_emb = mask  # 嵌入向量的掩码
        emb_flat = embeddings[mask_emb]
        # emb_flat = emb_real.reshape(-1, D)  # 得到展平的向量
        sim_matrix = torch.einsum('ik,jk->ij', emb_flat, emb_flat)  # 返回一个batch中所有向量的的相似度矩阵s_{ij}, [#total_term, #total_term]
        sim_removediag = remove_diagonal(sim_matrix)  #  去除了对角元的相似度矩阵[#total_term, #total_term-1]
        sim_logsoftmax = -F.log_softmax(sim_removediag/0.15, dim=-1)  # 得到每个锚点与所有的其他样本的logsoftmax加个负号，方便得到交叉熵

        batch_total_loss = torch.tensor(0.0)

        sample_idx:int = 0    # 起始样本索引
        for batch_idx in range(B):
            num_term_per_label = num_of_term_perlabel[batch_idx].item()   # 每个类别标签的中样本的个数
            sample_list = list(range(sample_idx,sample_idx+num_term_per_label))

            logsoftmax_subtensor = sim_logsoftmax[sample_list,sample_idx:sample_idx+num_term_per_label-1]    # 对本组样本有贡献的子矩阵
            sim_crossen = logsoftmax_subtensor.sum() / (num_term_per_label+1)  #得到本组样本交叉熵

            batch_total_loss = batch_total_loss + sim_crossen

            sample_idx = sample_idx + num_term_per_label   # 更新下一组标签样本起始索引


        return batch_total_loss / num_total_term if num_total_term > 0 else torch.tensor(0.0)   # 直接得到预训练中一个batch的loss
'''
loss_test = NTXentloss(temperature=0.1)
A = torch.tensor([[[0.92,0.2,(1-0.92**2-0.2**2)**0.5],[0.93,-0.18,(1-0.93**2-0.18**2)**0.5]],[[-0.91,0.19,(1-0.91**2-0.19**2)**0.5],[-0.94,-0.12,(1-0.94**2-0.12**2)**0.5]]])
B = torch.tensor([[[0.85,0.3,(1-0.85**2-0.3**2)**0.5],[0.86,-0.2,-(1-0.86**2-0.2**2)**0.5]],[[-0.87,0.13,(1-0.87**2-0.13**2)**0.5],[-0.90,-0.12,(1-0.90**2-0.12**2)**0.5]]])

loss1 = loss_test(A)
loss2 = loss_test(B)

print(loss1)
print(loss2)
'''