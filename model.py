import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

from positional_embeddings import SinusoidalEmbeddings, PositionalEmbeddings


# from lucidrains
# https://github.com/lucidrains/tf-bind-transformer/blob/main/tf_bind_transformer/tf_bind_transformer.py#L48
def fourier_encode(x, dims, theta=20000):
    device, dtype = x.device, x.dtype
    emb = math.log(theta) / (dims // 2)
    emb = torch.exp(torch.arange(dims // 2, device=device) * -emb)
    emb = rearrange(x, 'n -> n 1') * rearrange(emb, 'd -> 1 d')
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb


# from lucidrains
# https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py#L186

def exists(val):
    return val is not None


class AlibiPositionalBias(nn.Module):
    def __init__(self, heads, **kwargs):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> () h () ()')
        self.register_buffer('slopes', slopes, persistent=False)
        self.register_buffer('bias', None, persistent=False)

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                                                           :heads - closest_power_of_2]

    def forward(self, qk_dots):
        h, i, j, device = *qk_dots.shape[-3:], qk_dots.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return qk_dots + self.bias[..., :j]

        bias = torch.arange(j, device=device)
        bias = rearrange(bias, 'j -> () () () j')
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[1]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))

        self.register_buffer('bias', bias, persistent=False)
        return qk_dots + self.bias


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

        x = self.dropout(F.relu(self.fc_1(x)) ** 2)

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim: int, n_heads: int, dropout: float, device: torch.device,
                 relative_position_embedding: str, sequence_length: int, embedding_network_hidden: int = 256,
                 fourier_dims: int = 16) -> None:
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

        self.relative_position_embedding = relative_position_embedding

        self.fourier_dims = fourier_dims

        if relative_position_embedding in ["log_cpb", "linear_cpb"]:
            self.embedding_network: nn.Module = nn.Sequential(
                nn.Linear(in_features=1, out_features=embedding_network_hidden, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=embedding_network_hidden, out_features=n_heads, bias=True))

        elif relative_position_embedding == "fourier_cpb":
            self.embedding_network: nn.Module = nn.Sequential(
                nn.Linear(in_features=fourier_dims, out_features=embedding_network_hidden, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=embedding_network_hidden, out_features=n_heads, bias=True))
        else:
            self.embedding_network = None

        if relative_position_embedding == "alibi":
            self.alibi_pos_embedding = AlibiPositionalBias(n_heads)
        else:
            self.alibi_pos_embedding = None

        if relative_position_embedding == "rotary":
            self.rotary_pos_embedding = RotaryEmbedding(dim=(hid_dim // n_heads) // 2).to(device)
        else:
            self.rotary_pos_embedding = None

        self.sequence_length = sequence_length

        self.update_sizes(sequence_length)

    def forward(self, query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor,
                mask: Optional[torch.LongTensor] = None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        batch_size = query.shape[0]

        seq_len = query.shape[1]

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

        # if rotary:
        if self.rotary_pos_embedding:
            # apply the rotations to your queries and keys after the heads have been split out,
            # but prior to the dot product and subsequent softmax (attention)
            Q = self.rotary_pos_embedding.rotate_queries_or_keys(self.freqs, Q)
            K = self.rotary_pos_embedding.rotate_queries_or_keys(self.freqs, K)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        # if embedding network
        if self.embedding_network:
            # apply relative positional embeddings from SwinV2 (either log or linear or fourier)
            # TODO: fix context window length
            energy = energy + self.__get_relative_positional_encodings()[:, :, :seq_len, :seq_len]

        # if alibi pos embeddings
        if self.alibi_pos_embedding:
            # apply them to the dot product of Q and K
            energy = self.alibi_pos_embedding(energy)

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

    # from https://github.com/ChristophReich1996/Swin-Transformer-V2/blob/main/swin_transformer_v2/model_parts.py#L149
    def __make_relative_positions(self) -> None:
        """
        Method initializes the relative positions to compute the positional biases
        """
        indexes: torch.Tensor = torch.arange(self.sequence_length, device=self.scale.device)
        relative_indices: torch.Tensor = indexes[:, None] - indexes[None, :]
        relative_indices: torch.Tensor = relative_indices.reshape(-1, 1).float()
        if self.relative_position_embedding == "log_cpb":
            relative_indices_log: torch.Tensor = torch.sign(relative_indices) \
                                                 * torch.log(1. + relative_indices.abs())
            self.register_buffer("relative_indices", relative_indices_log)
        elif self.relative_position_embedding == "fourier_cpb":
            relative_indices = relative_indices.squeeze(1)
            relative_indices_fourier: torch.Tensor = fourier_encode(relative_indices, self.fourier_dims)
            self.register_buffer("relative_indices", relative_indices_fourier)
        else:
            self.register_buffer("relative_indices", relative_indices)

    def __get_relative_positional_encodings(self) -> torch.Tensor:
        """
        Method computes the relative positional encodings
        :return: (torch.Tensor) Relative positional encodings [1, number of heads, window size ** 2, window size ** 2]
        """
        relative_position_bias = self.embedding_network(self.relative_indices)
        relative_position_bias = relative_position_bias.permute(1, 0)
        relative_position_bias = relative_position_bias.reshape(self.n_heads, self.sequence_length,
                                                                self.sequence_length)
        return relative_position_bias.unsqueeze(0)

    def update_sizes(self, new_sequence_length: int) -> None:
        """
        Method updates the sequence length and so the relative positions
        :param new_sequence_length: (int) New sequence length
        """
        # Set new window size
        self.sequence_length = new_sequence_length

        if self.relative_position_embedding in ["log_cpb", "linear_cpb", "fourier_cpb"]:
            # Make new relative positions
            self.__make_relative_positions()



class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim: int,
                 n_heads: int,
                 pf_dim: int,
                 dropout: float,
                 device: torch.device,
                 relative_position_embedding: str,
                 sequence_length: int) -> None:
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device, relative_position_embedding,
                                                      sequence_length)
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

    def update_sizes(self, new_sequence_length: int) -> None:
        """
        Update the sequence length and thus relative positions
        :param new_sequence_length: int New sequence length
        """
        self.self_attention.update_sizes(new_sequence_length)


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
                 absolute_position_embedding: str = "sinusoidal",
                 relative_position_embedding: str = "log_cpb") -> None:
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
                                                  device,
                                                  relative_position_embedding,
                                                  max_length)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

        self.fc_out = nn.Linear(hid_dim, input_dim)

    def forward(self, src: torch.FloatTensor, src_mask: torch.FloatTensor, pos: torch.FloatTensor) -> torch.FloatTensor:
        # src = [batch size, src len]
        # src_mask = [batch size, 1, 1, src len]
        # pos = [batch size, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        src = (self.tok_embedding(src) * self.scale)

        if self.pos_embedding:
            src += self.pos_embedding(pos)

        src = self.dropout(src)

        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        # src = [batch size, src len, hid dim]

        src = self.fc_out(src)
        return src

    def update_sizes(self, new_sequence_length: int) -> None:
        """
        Update the sequence length and thus relative positions
        :param new_sequence_length: New sequence length
        """
        for layer in self.layers:
            layer.update_sizes(new_sequence_length)
