import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Callable
import numpy as np
import math

class DiversifiedRoPE(nn.Module):

    def __init__(self, d_model, num_heads, max_len=5000):
        super(DiversifiedRoPE, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0
        assert self.head_dim % 2 == 0
        dim = torch.arange(0, d_model, 2).float()
        theta = 1.0 / 10000 ** (dim / d_model)
        self.register_buffer('theta', theta)
        self.theta_slices = []
        theta_per_head = self.head_dim // 2
        for i in range(num_heads):
            start_idx = i * theta_per_head
            end_idx = (i + 1) * theta_per_head
            self.theta_slices.append((start_idx, end_idx))

    def apply_rotary_pos_emb(self, x):
        batch_size, seq_len, num_heads, head_dim = x.shape
        rotated_outputs = []
        for head_idx in range(num_heads):
            start_idx, end_idx = self.theta_slices[head_idx]
            theta_head = self.theta[start_idx:end_idx]
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
