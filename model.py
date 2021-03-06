import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

from absolute_positional_embeddings import SinusoidalEmbeddings, PositionalEmbeddings
from relative_positional_embeddings import AlibiPositionalBias


# from lucidrains
# https://github.com/lucidrains/tf-bind-transformer/blob/main/tf_bind_transformer/tf_bind_transformer.py#L48
def fourier_encode(x, dims, theta=20000):
    device, dtype = x.device, x.dtype
    emb = math.log(theta) / (dims // 2)
    emb = torch.exp(torch.arange(dims // 2, device=device) * -emb)
    emb = rearrange(x, 'n -> n 1') * rearrange(emb, 'd -> 1 d')
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb


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

        x = F.relu(self.fc_1(x)) ** 2

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        x = self.dropout(x)

        # x = [batch size, seq len, hid dim]

        return x


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim: int, n_heads: int, dropout: float, device: torch.device,
                 relative_position_embedding: str, sequence_length: int, embedding_network_hidden: int = 256,
                 fourier_dims: int = 16) -> None:
        super().__init__()

        self.should_use_cache = False
        self.cpb_cache = None
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

        elif relative_position_embedding == "linear_cpb_large":
            self.embedding_network: nn.Module = nn.Sequential(
                nn.Linear(in_features=1, out_features=embedding_network_hidden, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=embedding_network_hidden, out_features=embedding_network_hidden, bias=True),
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
            Q = self.rotary_pos_embedding.rotate_queries_or_keys(Q)
            K = self.rotary_pos_embedding.rotate_queries_or_keys(K)

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
        # skip computation and use cached values if activated
        if self.should_use_cache:
            return self.cpb_cache
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

        if self.relative_position_embedding in ["log_cpb", "linear_cpb", "fourier_cpb", "linear_cpb_large"]:
            # Make new relative positions
            self.__make_relative_positions()

            # if the cache is in use, update it with the new seqlen
            if self.should_use_cache:
                self.use_cache(self.should_use_cache)

    def use_cache(self, should_use: bool = False):
        if should_use:
            self.cpb_cache = self.__get_relative_positional_encodings()

        self.should_use_cache = should_use

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")

        # modify cache behavior whether in training or evaluation mode
        # cache is opposite of mode here
        # when mode == True, that is training mode, and so cache should be False/disabled
        # when mode == False, that is eval mode, and so cache should be True/enabled
        self.use_cache(not mode)

        self.training = mode
        for module in self.children():
            module.train(mode)
        return self


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

    def forward(self, src: torch.FloatTensor, src_mask: torch.FloatTensor) -> torch.FloatTensor:
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        # pre LN
        pre = self.self_attn_layer_norm(src)

        # self attention and residual
        src = src + self.self_attention(pre, pre, pre, src_mask)[0]

        # src = [batch size, src len, hid dim]

        # pre LN
        pre = self.ff_layer_norm(src)

        # positionwise feedforward and residual
        src = src + self.positionwise_feedforward(pre)

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
