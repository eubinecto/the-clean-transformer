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
                              padding="max_length",
                              # Do not add any special tokens
                              add_special_tokens=False,
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

    def __init__(self, tokenizer: BertTokenizer, max_length: int, start_token: str, device: torch.device):
        self.start_token = start_token
        super().__init__(tokenizer, max_length, device)

    def __call__(self, sents: List[str]):
        """
        :param sents:
        :return: (N, 2, L)
        """
        # "[SOS]" stands for  "Start of Sequence".
        encoded_l = self.encode([self.start_token + sent[:-1] for sent in sents])
        encoded_r = self.encode(sents)
        return torch.stack([encoded_l['input_ids'], encoded_r['input_ids']], dim=1).to(self.device)


def build_lookahead_mask(max_length: int, device: torch.device) -> torch.LongTensor:
    """
    build a look-ahead mask.
    :param max_length: L
    :param device:
    :return: lookahead_mask (L, L)
    """
    lookahead_mask = torch.tril(torch.ones(size=(max_length, max_length)), diagonal=0)
    return lookahead_mask.long().to(device)
