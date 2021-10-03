from typing import List, Tuple
import torch
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


def build_X(gibberish2kor: List[Tuple[str, str]], device: torch.device) -> torch.Tensor:
    """
    :param gibberish2kor:
    :param device:
    :return: X (N, L)
    """
    # TODO
    raise NotImplementedError


def build_Y(gibberish2kor: List[Tuple[str, str]], device: torch.device) -> torch.Tensor:
    """
    :param gibberish2kor:
    :param device:
    :return: Y (N, L)
    """
    # TODO
    raise NotImplementedError


def build_M(gibberish2kor: List[Tuple[str, str]], device: torch.device) -> torch.Tensor:
    """
    :param gibberish2kor:
    :param device:
    :return: M (N, L, L)  - 3차원?
    """
    # TODO - use torch.triu?
    raise NotImplementedError
