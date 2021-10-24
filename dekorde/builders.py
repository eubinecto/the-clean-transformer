from typing import List
import torch
from transformers import BertTokenizer, BatchEncoding


class TensorBuilder:
    def __init__(self, tokenizer: BertTokenizer, max_length: int, device: torch.device):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __call__(self, sents: List[str]) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, sents: List[str]) -> BatchEncoding:
        return self.tokenizer(text=sents,
                              truncation=True,
                              padding=True,
                              max_length=self.max_length,
                              return_tensors="pt")


class XBuilder(TensorBuilder):

    def __call__(self, sents: List[str]) -> torch.Tensor:
        """
        :param sents:
        :return: (N, 2, L)
        """
        encoded = self.encode(sents)
        return torch.stack([encoded['input_ids'],
                            encoded['attention_mask']], dim=1).to(self.device)


class YBuilder(TensorBuilder):
    def __call__(self, sents: List[str]):
        """
        :param sents:
        :return: (N, 2, 2, L)
        """
        # "s" stands for  "start of sequence".
        encoded_l = self.encode(["s" + sent[:-1] for sent in sents])
        encoded_r = self.encode(sents)
        Y_l = torch.stack([encoded_l['input_ids'],
                           encoded_l['attention_mask']], dim=1)
        Y_2 = torch.stack([encoded_r['input_ids'],
                           encoded_r['attention_mask']], dim=1)
        return torch.stack([Y_l, Y_2], dim=1).to(self.device)


def build_lookahead_mask(max_length: int, device: torch.device) -> torch.LongTensor:
    """
    build a look-ahead mask.
    :param max_length: L
    :param device:
    :return: lookahead_mask (L, L)
    """
    lookahead_mask = torch.tril(torch.ones(size=(max_length, max_length)), diagonal=0)
    return lookahead_mask.long().to(device)
