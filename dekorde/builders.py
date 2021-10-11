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


def build_X(gibs: List[str], tokenizer: Tokenizer, max_length: int, device: torch.device) -> torch.LongTensor:
    return build_I(gibs, tokenizer, max_length, device)


def build_Y(kors: List[str], tokenizer: Tokenizer, max_length: int, device: torch.device) -> torch.LongTensor:
    Y_l = build_I(["s" + kor[:-1] for kor in kors], tokenizer, max_length, device)
    Y_r = build_I(kors, tokenizer, max_length, device)
    return torch.stack([Y_l, Y_r], dim=1).long()



def build_lookahead_mask(max_length: int, device: torch.device) -> torch.LongTensor:
    """
    build a look-ahead mask.
    :param max_length: L
    :param device:
    :return: lookahead_mask (L, L)
    """
    lookahead_mask = torch.tril(torch.ones(size=(max_length, max_length)), diagonal=0)
    return lookahead_mask.long().to(device)
