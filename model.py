from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from positional_embeddings import SinusoidalEmbeddings, PositionalEmbeddings


# transformer model based on Encoder part of this implementation:
# https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
# by Ben Trevett, MIT licensed


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim: int, pf_dim: int, dropout: float) -> None:
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # x = [batch size, seq len, hid dim]

        x = self.dropout(F.gelu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim: int, n_heads: int, dropout: float, device: torch.device) -> None:
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor,
                mask: Optional[torch.LongTensor] = None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim: int,
                 n_heads: int,
                 pf_dim: int,
                 dropout: float,
                 device: torch.device) -> None:
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.FloatTensor, src_mask: torch.FloatTensor) -> torch.FloatTensor:
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hid_dim: int,
                 n_layers: int,
                 n_heads: int,
                 pf_dim: int,
                 dropout: float,
                 device: torch.device,
                 max_length: int = 128,
                 absolute_position_embedding: str = "sinusoidal") -> None:
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)

        if absolute_position_embedding == "sinusoidal":
            self.pos_embedding = SinusoidalEmbeddings(max_length, hid_dim)
        elif absolute_position_embedding == "scaled_sinusoidal":
            self.pos_embedding = SinusoidalEmbeddings(max_length, hid_dim, learnable_scaling=True)
        elif absolute_position_embedding == "learned":
            self.pos_embedding = PositionalEmbeddings(max_length, hid_dim)
        elif absolute_position_embedding == "none":
            self.pos_embedding = None

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src: torch.FloatTensor, src_mask: torch.FloatTensor) -> torch.FloatTensor:
        # src = [batch size, src len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, src len]

        src = (self.tok_embedding(src) * self.scale)

        if self.pos_embedding:
            src += self.pos_embedding(pos)

        src = self.dropout(src)

        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        # src = [batch size, src len, hid dim]

        return src
