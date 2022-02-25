from typing import Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional


# Absolute Positional Embeddings:

# Sinusoidal Embeddings:

# based on:
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/4f4a192f0fd272102c8852b00b1007dffd292b90/transformer/Models.py#L11
# MIT Licensed by Yu-Hsiang Huang

# Learned Scaling from "Transformer Quality in Linear Time" arxiv preprint: https://arxiv.org/abs/2202.10447
# Appendix D. Code 2 contains tensorflow pseudocode for learned scaling of sinusoidal embeddings

class SinusoidalEmbeddings(nn.Module):

    def __init__(self, n_position: int, hidden_width: int, learnable_scaling: bool = False, padding_idx: Optional[int] = None) -> None:
        """

        :param n_position: max number of positions
        :param hidden_width: width of output embeddings, usually the same as model width
        :param learnable_scaling: whether to add a learnable scaling to the embeddings, like in the arxiv preprint
        https://arxiv.org/abs/2202.10447 "Transformer Quality in Linear Time"
        :param padding_idx: padding index for embedding, ignores gradient for this index, default None
        """
        super().__init__()

        self.embedding = nn.Embedding(n_position, hidden_width, padding_idx=padding_idx)
        self.embedding.weight.data = self.position_encoding_init(n_position, hidden_width)

        self.embedding.weight.requires_grad = False

        if learnable_scaling:
            self.scaling = nn.Parameter(torch.FloatTensor([1 / hidden_width ** 0.5]), requires_grad=True)
        else:
            self.scaling = 1

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        Embeds a tensor of indices into an absolute sinusoidal embeddings, optionally with a learned scaling factor
        :param x: input LongTensor, expected to be tensor of arbitrary shape containing indices of shape (*)
        :return: return FloatTensor of shape (*, H) where * is input shape, embedded version of indices
        """
        return self.embedding(x) * self.scaling

    @staticmethod
    def position_encoding_init(n_position: int, hidden_width: int) -> torch.FloatTensor:
        """
        Generates sinusoidal embedding matrix from input parameters (sequence length and sequence width)
        :param n_position: max number of input positions
        :param hidden_width: width of the embeddings, usually width of transformer model
        :return:
        """
        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / hidden_width) for i in range(hidden_width)]
            if pos != 0 else np.zeros(hidden_width) for pos in range(n_position)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        return torch.from_numpy(position_enc).type(torch.FloatTensor)


# Learned Positional Embeddings

class PositionalEmbeddings(nn.Module):

    def __init__(self, n_position: int, hidden_width: int, padding_idx: Optional[int] = None) -> None:
        """
        An absolute learned positional embedding, where each position is learned and fixed.
        :param n_position: maximum number of positions, max position index + 1 (0 is an index)
        :param hidden_width: width of the embeddings, usually width of transformer model
        :param padding_idx: padding index, no gradients calculated on this index, default is None
        """
        super().__init__()

        self.embedding = nn.Embedding(n_position, hidden_width, padding_idx=padding_idx)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Embeds tensor of indices into tensor of embeddings
         :param x: input LongTensor, expected to be tensor of arbitrary shape containing indices of shape (*)
        :return: return FloatTensor of shape (*, H) where * is input shape, embedded version of indices
        """
        return self.embedding(x)


