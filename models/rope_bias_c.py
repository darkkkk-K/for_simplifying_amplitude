import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FlexibleConstrainedLearnableRoPE(nn.Module):

    def __init__(self, d_model, num_heads, max_len=5000, use_shared_theta=True, theta_init_method='original', min_freq=1e-06, max_freq=1.0, use_log_space=True):
        """
        灵活带约束的可学习RoPE

        Args:
            d_model: 模型隐藏维度
            num_heads: 注意力头数
            max_len: 最大序列长度
            use_shared_theta: True=所有头共享θ, False=每个头独立θ
            theta_init_method: 初始化方法 ("original", "random", "log_uniform")
            min_freq: 最小频率约束
            max_freq: 最大频率约束
            use_log_space: 是否在log空间优化
        """
        super(FlexibleConstrainedLearnableRoPE, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_shared_theta = use_shared_theta
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.use_log_space = use_log_space
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        assert self.head_dim % 2 == 0, 'head_dim must be even'
        if use_shared_theta:
            param_shape = (d_model // 2,)
        else:
            param_shape = (num_heads, self.head_dim // 2)
        if use_log_space:
            self.log_theta = nn.Parameter(torch.zeros(param_shape))
            self._init_log_theta(theta_init_method)
        else:
            self.theta = nn.Parameter(torch.zeros(param_shape))
            self._init_theta(theta_init_method)

    def _init_theta(self, method):
        """直接初始化θ参数"""
        if method == 'original':
            if self.use_shared_theta:
                dim_indices = torch.arange(0, self.d_model, 2).float()
                theta_init = 1.0 / 10000 ** (dim_indices / self.d_model)
                self.theta.data = theta_init
            else:
                for head_idx in range(self.num_heads):
                    start_idx = head_idx * (self.head_dim // 2)
                    end_idx = (head_idx + 1) * (self.head_dim // 2)
                    dim_indices = torch.arange(start_idx, end_idx).float()
                    theta_init = 1.0 / 10000 ** (dim_indices / self.d_model)
                    self.theta.data[head_idx] = theta_init
        elif method == 'random':
            nn.init.uniform_(self.theta, 0.001, 0.1)
        elif method == 'log_uniform':
            log_theta = torch.rand_like(self.theta) * 8 - 4
            self.theta.data = torch.exp(log_theta)

    def _init_log_theta(self, method):
        """初始化log空间θ参数"""
        if method == 'original':
            if self.use_shared_theta:
                dim_indices = torch.arange(0, self.d_model, 2).float()
                theta_init = 1.0 / 10000 ** (dim_indices / self.d_model)
                theta_init = torch.clamp(theta_init, min=1e-08)
                self.log_theta.data = torch.log(theta_init)
            else:
                for head_idx in range(self.num_heads):
                    start_idx = head_idx * (self.head_dim // 2)
                    end_idx = (head_idx + 1) * (self.head_dim // 2)
                    dim_indices = torch.arange(start_idx, end_idx).float()
                    theta_init = 1.0 / 10000 ** (dim_indices / self.d_model)
                    theta_init = torch.clamp(theta_init, min=1e-08)
                    self.log_theta.data[head_idx] = torch.log(theta_init)
        elif method == 'random':
            log_min = math.log(self.min_freq)
            log_max = math.log(self.max_freq)
            nn.init.uniform_(self.log_theta, log_min, log_max)
        elif method == 'log_uniform':
            log_min = math.log(0.0001)
            log_max = math.log(1.0)
            nn.init.uniform_(self.log_theta, log_min, log_max)

    @property
    def constrained_theta(self):
        """获取带约束的θ值"""
        if self.use_log_space:
            theta = torch.exp(self.log_theta)
            theta = torch.clamp(theta, self.min_freq, self.max_freq)
            return theta
        else:
            theta = F.softplus(self.theta)
            theta = torch.clamp(theta, self.min_freq, self.max_freq)
            return theta

    def get_theta_for_head(self, head_idx=None):
        """
        获取指定头的θ参数

        Args:
            head_idx: 头索引，None表示获取所有
        """
        theta = self.constrained_theta
        if self.use_shared_theta:
            if head_idx is not None:
                start_idx = head_idx * (self.head_dim // 2)
                end_idx = (head_idx + 1) * (self.head_dim // 2)
                return theta[start_idx:end_idx]
            else:
                return theta
        elif head_idx is not None:
            return theta[head_idx]
        else:
            return theta

    def apply_rotary_pos_emb(self, x):
        """
        应用旋转位置编码

        Args:
            x: 输入张量 (batch_size, seq_len, num_heads, head_dim)
        """
        batch_size, seq_len, num_heads, head_dim = x.shape
        if self.use_shared_theta:
            theta_used = self.get_theta_for_head()[:head_dim // 2]
            positions = torch.arange(seq_len, device=x.device).float()
            angles = positions.unsqueeze(1) * theta_used.unsqueeze(0)
            cos_vals = torch.cos(angles).unsqueeze(0).unsqueeze(2)
            sin_vals = torch.sin(angles).unsqueeze(0).unsqueeze(2)
            x_reshaped = x.view(batch_size, seq_len, num_heads, head_dim // 2, 2)
            x1 = x_reshaped[..., 0]
            x2 = x_reshaped[..., 1]
            x1_rot = x1 * cos_vals - x2 * sin_vals
            x2_rot = x1 * sin_vals + x2 * cos_vals
            x_rotated = torch.stack([x1_rot, x2_rot], dim=-1)
            x_rotated = x_rotated.view(batch_size, seq_len, num_heads, head_dim)
            return x_rotated
        else:
            rotated_outputs = []
            for head_idx in range(num_heads):
                theta_head = self.get_theta_for_head(head_idx)
                x_head = x[:, :, head_idx:head_idx + 1, :]
                positions = torch.arange(seq_len, device=x.device).float()
                angles = positions.unsqueeze(1) * theta_head.unsqueeze(0)
                cos_vals = torch.cos(angles).unsqueeze(0).unsqueeze(2)
                sin_vals = torch.sin(angles).unsqueeze(0).unsqueeze(2)
                x_head_reshaped = x_head.view(batch_size, seq_len, 1, head_dim // 2, 2)
                x1 = x_head_reshaped[..., 0]
                x2 = x_head_reshaped[..., 1]
                x1_rot = x1 * cos_vals - x2 * sin_vals
                x2_rot = x1 * sin_vals + x2 * cos_vals
                x_head_rotated = torch.stack([x1_rot, x2_rot], dim=-1)
                x_head_rotated = x_head_rotated.view(batch_size, seq_len, 1, head_dim)
                rotated_outputs.append(x_head_rotated)
            return torch.cat(rotated_outputs, dim=2)

    def extra_repr(self):
        """显示模块配置信息"""
        return f'd_model={self.d_model}, num_heads={self.num_heads}, use_shared_theta={self.use_shared_theta}, min_freq={self.min_freq:.2e}, max_freq={self.max_freq:.2f}, use_log_space={self.use_log_space}'
