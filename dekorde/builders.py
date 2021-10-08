from typing import List, Tuple
import torch
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


# this is for building an input
def build_I(sents: List[str], tokenizer: Tokenizer, max_length: int, device: torch.device) -> torch.LongTensor:
    seqs = tokenizer.texts_to_sequences(texts=sents)
    # use zero for the pad token
    seqs = pad_sequences(sequences=seqs, maxlen=max_length, padding="post", value=0)
    return torch.LongTensor(seqs).to(device)


def build_M(data_size: int, max_length: int, device: torch.device) -> torch.Tensor:
    """
    :param data_size: N
    :param max_length: L
    :param device:
    :return: M (N, L, L)  - 3차원?
    """
    M = torch.tril(torch.ones(size=(max_length, max_length)), diagonal=0)\
             .expand(data_size, max_length, max_length)
    return M.to(device)