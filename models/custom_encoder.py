import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .rope_bias_c import FlexibleConstrainedLearnableRoPE
from .rope_bias_a import RoPE_Encoding
from .rope_bias_b import DiversifiedRoPE
from typing import List, Tuple, Optional, Dict, Any, Callable


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError(f"Activation should be relu/gelu, not {activation}")


class CustomMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, batch_first=True):
        super(CustomMultiheadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        # QKV投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.last_pos_bias = nn.Parameter(torch.tensor(2.0))

    def forward(self,
                query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, key_padding_mask:torch.Tensor,
                apply_attention_bias_func, rope:FlexibleConstrainedLearnableRoPE, attn_mask:torch.Tensor=None):
        '''
        rope:旋转编码实例。
        '''
        batch_size = query.size(0)

        # 投影Q, K, V
        q = self.q_proj(query)  # (batch_size, seq_len, d_model)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 重塑为多头格式
        q = q.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)  # [B, nhead, seq_len, head_dim]
        k = k.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)

        # 应用RoPE
        if rope is not None:
            q = rope.apply_rotary_pos_emb(q.transpose(1, 2)).transpose(1, 2)   # apply_rotary_pos_emb方法是
            k = rope.apply_rotary_pos_emb(k.transpose(1, 2)).transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, nhead, seq_len, seq_len]


        # 应用注意力偏置
        if apply_attention_bias_func is not None:
            attn_scores_bias = self._apply_attention_bias(attention_scores=attn_scores, padding_mask=key_padding_mask, if_learn=False)
        else:
            attn_scores_bias = attn_scores
        # 应用注意力padding掩码
        if attn_mask is not None:
            attn_scores_bias += attn_mask
        if key_padding_mask is not None:
            attn_scores_bias = attn_scores_bias.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
            all_padding_mask = key_padding_mask.all(dim=1)  # (batch_size,)

            # 对于全掩码序列，将注意力分数设置为0（而不是-inf）
            if all_padding_mask.any():
                # 创建一个掩码来标识全零序列
                full_zero_mask = all_padding_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                full_zero_mask = full_zero_mask.expand_as(attn_scores_bias)

                # 将全零序列的注意力分数设置为一个很小的负数
                # 这样softmax后会得到均匀分布，而不是NaN
                attn_scores_bias = torch.where(full_zero_mask,
                                        torch.tensor(-1e-6, device=attn_scores_bias.device),
                                        attn_scores_bias)

        # 计算注意力权重和输出
        attn_weights = F.softmax(attn_scores_bias, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # [B, nhead, seq_len, head_dim]

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )    # [B, seq_len, D], contiguous管理内存

        # 输出投影
        attn_output = self.out_proj(attn_output)

        return attn_output

    def _apply_attention_bias(self, attention_scores:torch.Tensor, padding_mask:torch.Tensor, if_learn:bool = False):
        '''
        由于最后两个token的信息非常重要，因此在这里加入注意力偏置，src_mask[B,T]
        '''
        B, nhead, sqlen, dim = attention_scores.shape
        # padding_mask=True 表示 padding；最后两枚“有效 token”应由非 padding
        # 的数量确定。旧实现统计了 padding 数，导致同一项仅因与更长样本组 batch
        # 就被施加到不同位置的 bias，冻结 encoder 的输出因而依赖 batch 组成。
        valid_lengths = (~padding_mask.bool()).sum(dim=-1)
        bias_matrix = torch.zeros_like(attention_scores)  # attention score的形状为[B, num_head, seq_len, seq_len]注意力分数矩阵
        bias_value = self.last_pos_bias if if_learn else self.last_pos_bias.detach()
        for i in range(B):
            end = int(valid_lengths[i].item())
            if end > 0:
                start = max(0, end - 2)
                bias_matrix[i, :, start:end, :] = bias_value
                bias_matrix[i, :, :, start:end] = bias_value
        return attention_scores + bias_matrix



class BatchFirstTransformerEncoderLayerWithCustomAttention(nn.Module):
    def __init__(self, d_model, nhead, rope:FlexibleConstrainedLearnableRoPE, dim_feedforward=3072, dropout=0.1,
                 activation="relu", apply_attention_bias_func="OK"):
        super(BatchFirstTransformerEncoderLayerWithCustomAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead

        # 使用自定义的注意力机制
        self.self_attn = CustomMultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 激活函数
        self.activation = _get_activation_fn(activation)

        # RoPE和注意力偏置函数
        self.rope = rope
        self.apply_attention_bias_func = apply_attention_bias_func

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 归一化
        src2 = self.norm1(src)
        #print(src2)
        # 使用自定义注意力
        attn_output = self.self_attn(
            query=src2,
            key=src2,
            value=src2,
            key_padding_mask=src_key_padding_mask,
            apply_attention_bias_func=self.apply_attention_bias_func,
            attn_mask=src_mask,
            rope=self.rope,
        )
        # 残差连接和dropout

        attn_output = self.dropout1(attn_output)
        src = src + attn_output

        # 前馈网络部分
        src2 = self.norm2(src)
        #print(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src2 = self.dropout2(src2)
        src = src + src2
        return src

class BatchFirstTransformerEncoder(nn.Module):
    '''
    RoPE = 'shared_learnable', 'unshared_learnable', 'shared_fixed', 'unshared_fixed'
    '''
    def __init__(self, encoder_layer:BatchFirstTransformerEncoderLayerWithCustomAttention,
                 d_model, fnn_dim, num_head, num_layers, norm=None, RoPE_mode:str='shared_learnable'):
        super().__init__()
        if RoPE_mode == 'shared_learnable':
            posenc = FlexibleConstrainedLearnableRoPE(
            d_model=d_model,
            num_heads=num_head,
            use_shared_theta=True
            )
        elif RoPE_mode == 'unshared_learnable':
            posenc = FlexibleConstrainedLearnableRoPE(
            d_model=d_model,
            num_heads=num_head,
            use_shared_theta=False
            )
        elif RoPE_mode == 'shared_fixed':
            posenc = RoPE_Encoding(
            emb_dim=d_model
            )
        elif RoPE_mode == 'unshared_fixed':
            posenc = DiversifiedRoPE(
            d_model=d_model,
            num_heads=num_head,
            )

        self.layers = nn.ModuleList([encoder_layer(d_model=d_model,nhead=num_head,rope = posenc,dim_feedforward = fnn_dim) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.num = sum(p.numel() for p in self.layers.parameters())
    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        src: (batch_size, seq_len, d_model)
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
