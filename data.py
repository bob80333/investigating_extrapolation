import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from tqdm import tqdm


class TextDataset(Dataset):

    def __init__(self, text_files: List[Path], tokenizer_path: str, sequence_length: int = 128, stride: int = 128, pretokenized: bool = False):
        """
        A class that holds a basic text dataset in memory in tokenized form, along with the tokenizer
        :param text_files: list of paths to the various text files/documents to use as data
        :param tokenizer_path: path to huggingface Tokenizers json
        :param sequence_length: length of sequences to return
        :param stride: gap between sequences
        """
        super().__init__()

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.text_files = text_files

        if stride == 0:
            raise ValueError("Stride must be >= 1, otherwise the same piece of data will be repeated infinite times")

        self.encoded_tokens = []
        self.n_tokens_windows = []

        self.length = 0
        self.sequence_length = sequence_length
        self.stride = stride

        total_tokens = 0

        for file in tqdm(text_files):
            if not pretokenized:
                with open(file, 'r', encoding='utf-8') as reader:
                    text = reader.read()

                # add SOS and EOS tokens to each document
                text = "<SOS>" + text + "<EOS>"

                # encode into tokens
                ids = self.tokenizer.encode(text).ids

            else:
                with open(file, 'rb') as reader:
                    ids = np.array(pickle.load(reader))
            # store tokens
            self.encoded_tokens.append(ids)
            total_tokens += len(ids)

            # store number of possible windows for this file, this is for presenting multiple files as one whole
            # subtract 1 window, for cases of small stride (e.g. stride=1)
            n_windows = ((len(ids) - sequence_length) // stride)
            self.n_tokens_windows.append(n_windows)
            self.length += n_windows

        print("Loaded dataset of", total_tokens, "tokens")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.LongTensor:
        for idx, n_windows in enumerate(self.n_tokens_windows):
            if index < n_windows:  # the index is within this window if it is less than the n_windows
                token_idx = index * self.stride
                return torch.LongTensor(self.encoded_tokens[idx][token_idx:token_idx + self.sequence_length])
            else:
                index -= n_windows  # subtract this windowing, move to the next

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)
