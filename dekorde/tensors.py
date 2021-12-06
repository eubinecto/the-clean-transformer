from typing import List
import torch
from transformers import BertTokenizer, BatchEncoding


class TensorBuilder:
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class DataBuilder(TensorBuilder):
    def __init__(self, tokenizer: BertTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, sents: List[str]) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, sents: List[str]) -> BatchEncoding:
        # you need to do it your way... because you have to add ...
        # wait, you can, just.. use... cls & sep.
        return self.tokenizer(text=sents,
                              truncation=True,
                              # to fix the maximum length
                              padding="max_length",
                              # Do not add any special tokens
                              add_special_tokens=False,
                              max_length=self.max_length,
                              return_tensors="pt")


class InputsBuilder(DataBuilder):

    def __call__(self, sents: List[str]) -> torch.Tensor:
        """
        :param sents:
        :return: (N, 2, L) - input_ids & attention_mask
        """
        sents = [
            self.tokenizer.bos_token + " " + sent + " " + self.tokenizer.eos_token
            for sent in sents
        ]
        encoded = self.encode(sents)
        return torch.stack([encoded['input_ids'],
                            encoded['attention_mask']], dim=1)


class TargetsBuilder(DataBuilder):

    def __init__(self, tokenizer: BertTokenizer, max_length: int):
        super().__init__(tokenizer, max_length)

    def __call__(self, sents: List[str]):
        """
        :param sents:
        :return: (N, 2, 2, L)
        """
        # adds BOS token at the beginning, ends with the second last token
        encoded = self.encode([self.tokenizer.bos_token + " " + sent for sent in sents])
        targets = torch.stack([encoded['input_ids'], encoded['attention_mask']], dim=1)
        # begins with the second token, adds EOS token at the end
        encoded_y = self.encode([sent + " " + self.tokenizer.eos_token for sent in sents])
        targets_y = torch.stack([encoded_y['input_ids'], encoded_y['attention_mask']], dim=1)
        return torch.stack([targets, targets_y], dim=1)


class LookAheadMaskBuilder(TensorBuilder):

    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self):
        lookahead_mask = torch.tril(torch.ones(size=(self.max_length, self.max_length)), diagonal=0)
        return lookahead_mask.long()
