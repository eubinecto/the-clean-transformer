from typing import List
import torch
from transformers import BertTokenizer, BatchEncoding


class DataBuilder:
    def __init__(self, tokenizer: BertTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, *args) -> torch.Tensor:
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


class TrainInputsBuilder(DataBuilder):

    def __call__(self, srcs: List[str], tgts: List[str]) -> torch.Tensor:
        """
        :param srcs:
        :param tgts:
        :return: (N, 2, 2, L) - input_ids & attention_mask
        """
        # the source sentences, which are to be fed as the inputs to the encoder
        encoded_src = self.encode([
            self.tokenizer.bos_token + " " + sent + " " + self.tokenizer.eos_token
            for sent in srcs
        ])
        # the target sentences, which are to be fed as the inputs to the decoder
        encoded_tgt = self.encode([
            # starts with bos, but does not end with eos (left-shifted)
            self.tokenizer.bos_token + " " + sent
            for sent in tgts
        ])
        inputs_src = torch.stack([encoded_src['input_ids'],
                                  encoded_src['attention_mask']], dim=1)
        inputs_tgt = torch.stack([encoded_tgt['input_ids'],
                                  encoded_tgt['attention_mask']], dim=1)
        inputs = torch.stack([inputs_src, inputs_tgt], dim=1)
        return inputs


class InferInputsBuilder(DataBuilder):
    def __call__(self, srcs: List[str]) -> torch.Tensor:
        """
        :param srcs:
        :return: (N, 2, L)
        """
        encoded_src = self.encode([
            self.tokenizer.bos_token + " " + sent + " " + self.tokenizer.eos_token
            for sent in srcs
        ])
        encoded_tgt = self.encode([
            # just start with bos token
            self.tokenizer.bos_token
            for _ in srcs
        ])
        inputs_src = torch.stack([encoded_src['input_ids'],
                                  encoded_src['attention_mask']], dim=1)
        inputs_tgt = torch.stack([encoded_tgt['input_ids'],
                                  encoded_tgt['attention_mask']], dim=1)
        inputs = torch.stack([inputs_src, inputs_tgt], dim=1)
        return inputs


class LabelsBuilder(DataBuilder):

    def __init__(self, tokenizer: BertTokenizer, max_length: int):
        super().__init__(tokenizer, max_length)

    def __call__(self, tgts: List[str]):
        """
        :param tgts:
        :return: (N, L)
        """
        # to be used as the labels
        encoded = self.encode(
            # does not start with eos, but ends with eos.
            [sent + " " + self.tokenizer.eos_token for sent in tgts]
        )
        label_ids = encoded['input_ids']
        return label_ids
