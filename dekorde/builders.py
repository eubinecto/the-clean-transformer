from typing import List, Tuple
import torch
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


def build_I(sents: List[str],
            tokenizer: Tokenizer,
            max_length: int,
            device: torch.device) -> torch.Tensor:
    """
    This function converts raw human readable sentences to machine readable sequences
    Keras Text preprocessing module includes Tokenizer class that helps ease of converting text to sequence.

    :param sents:
    :param tokenizer:
    :param max_length:
    :param device:
    :return:
    """
    return torch.LongTensor(
        # padding 추가 (뒤에)
        pad_sequences(
            sequences=tokenizer.texts_to_sequences(texts=sents),
            maxlen=max_length,      # 시퀀스의 최대 길이
            padding="post",         # padding 의 위치는?            뒤에
            value=0                 # padding 은 어떤값으로 처리?     0
        )
    ).to(device)


def build_M(sents: List[str],
            head_size: int,
            max_sentence_length: int,
            device: torch.device) -> torch.Tensor:
    """
    how to mask
    :param head_size:
    :param sents:
    :param max_sentence_length:
    :param device:
    :return: M (N, L, L)  - 3차원?
    """

    M = torch.tril(torch.ones(max_sentence_length, max_sentence_length)).repeat(len(sents), head_size, 1, 1)
    return M.to(device)
