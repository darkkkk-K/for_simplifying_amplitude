import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Callable
import numpy as np
import math
'''
所有注意力头共享一组旋转参数
'''
class RoPE_Encoding(nn.Module):
    def __init__(self, emb_dim=768, dropout=0.1, max_len=5000):
        super(RoPE_Encoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = emb_dim
        self.max_len = max_len

        # 计算旋转角度
        theta = 1.0 / (10000**(torch.arange(0, emb_dim, 2).float()/emb_dim)) # [D/2]
        self.register_buffer('theta', theta)  # 进入缓存区，不参与梯度下降

    def apply_rotary_pos_emb(self, x:torch.Tensor):
        """
        x: (batch_size, seq_len, num_heads, head_dim)
        返回: 应用了旋转位置编码的x, 这里使用局部theta参数，所有头都使用theta前D/(2*num_head)个
        """
        B, seq_len, num_head, head_dim = x.shape
        assert head_dim % 2 == 0  # 确保注意力头维度是偶数

        # 创建位置索引
        positions = torch.arange(seq_len, device=x.device).float().view(1, seq_len, 1, 1)  # [1, seq_len, 1, 1]
        theta_used = self.theta[:head_dim//2]  # [D/(2*num_head)]
        # 计算旋转角度: (1, seq_len, 1, head_dim//2)
        angles = positions * theta_used.view(1, 1, 1, -1)

        # 计算cos和sin
        cos_vals = torch.cos(angles)  # (1, seq_len, 1, head_dim//2)
        sin_vals = torch.sin(angles)  # (1, seq_len, 1, head_dim//2)

        x_head_reshaped = x.view(B, seq_len, num_head, head_dim//2, 2)
        x1 = x_head_reshaped[..., 0]
        x2 = x_head_reshaped[..., 1]

        x1_rot = x1 * cos_vals - x2 * sin_vals
        x2_rot = x1 * sin_vals + x2 * cos_vals

        # 重新组合
        x_head_rotated = torch.stack([x1_rot, x2_rot], dim=-1)  # dim = -1
        x_rotated = x_head_rotated.view(B, seq_len, num_head, head_dim)

        return x_rotated   # [batch_size, seq_len, num_heads, head_dim]