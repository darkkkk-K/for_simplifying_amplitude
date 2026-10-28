"""Pretrained three-layer RoPE Transformer encoder."""
import torch
from torch import nn
from torch.nn import functional as F
from .custom_encoder import BatchFirstTransformerEncoderLayerWithCustomAttention, BatchFirstTransformerEncoder
class ClusteringMachine1_RoPE(nn.Module):
    """该class需要自定义QKV注意力矩阵，加入了RoPE位置编码和最后两个位置的注意力偏置"""

    def __init__(self, vocab_size, emb_dim=768, numhead=8, numenc1=4, fnn_dim=3072, dropout=0.1, RoPE_mode: str='shared_learnable'):
        super(ClusteringMachine1_RoPE, self).__init__()
        self.emb_dim = emb_dim
        self.embdding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.transformer = BatchFirstTransformerEncoder(encoder_layer=BatchFirstTransformerEncoderLayerWithCustomAttention, d_model=emb_dim, fnn_dim=fnn_dim, num_head=numhead, num_layers=numenc1, RoPE_mode=RoPE_mode)
        self.projector = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))

    def forward(self, src: torch.Tensor, RL: bool=False):
        """
        需要得到除了合成的项向量之外，还需要得到离合成向量最近的头两个spinor bracket。
        """
        batch_size, set_size, seq_len = src.shape
        src_re = src.reshape(batch_size * set_size, seq_len)
        padding_mask = src_re == 0
        src_emb = self.embdding(src_re)
        output_emb = self.transformer(src=src_emb, src_key_padding_mask=padding_mask)
        output = self.projector(output_emb)
        mask = (src_re != 0).float()
        if RL:
            output_for_RL = output * mask.unsqueeze(-1)
            return output_for_RL
        out = (output * mask.unsqueeze(-1)).sum(dim=1)
        output_norm = F.normalize(out, dim=-1)
        final_out = output_norm.reshape(batch_size, set_size, self.emb_dim)
        return final_out
