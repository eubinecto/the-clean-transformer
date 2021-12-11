import torch
from abc import ABC
from typing import List, Tuple
from tokenizers import Tokenizer, Encoding


class DataBuilder:
    def __init__(self, tokenizer: Tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, sents: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # we use enable padding here, rather than on creating the tokenizer,
        # to set maximum length on encoding
        self.tokenizer.enable_padding(pad_token=self.tokenizer.pad_token,  # noqa
                                      pad_id=self.tokenizer.pad_token_id,  # noqa
                                      length=self.max_length)
        # don't add special tokens, we will add them ourselves
        encodings: List[Encoding] = self.tokenizer.encode_batch(sents, add_special_tokens=False)
        ids = torch.LongTensor([encoding.ids for encoding in encodings])
        # 1's: non-pad tokens. 0's: padded tokens
        key_padding_mask = torch.BoolTensor([encoding.attention_mask for encoding in encodings])
        return ids, key_padding_mask


class InputsBuilder(DataBuilder, ABC):

    def src_inputs(self, srcs: List[str]) -> torch.Tensor:
        # the source sentences, which are to be fed as the inputs to the encoder
        src_ids, src_key_padding_mask = self.encode([
            self.tokenizer.bos_token + " " + sent + " " + self.tokenizer.eos_token  # noqa
            for sent in srcs
        ])
        src_inputs = torch.stack([src_ids, src_key_padding_mask], dim=1)  # (N, 2, L)
        return src_inputs

    def tgt_inputs(self, tgts: List[str]) -> torch.Tensor:
        raise NotImplementedError


class TrainInputsBuilder(InputsBuilder):

    def __call__(self, srcs: List[str], tgts: List[str]) -> torch.Tensor:
        assert len(srcs) == len(tgts)
        src_inputs = self.src_inputs(srcs)  # (N, 2, L)
        tgt_inputs = self.tgt_inputs(tgts)  # (N, 2, L)
        inputs = torch.stack([src_inputs, tgt_inputs], dim=1)  # (N, 2, 2, L)
        return inputs

    def tgt_inputs(self, tgts: List[str]) -> torch.Tensor:
        """
        :param tgts:
        :return: (N, 2, 2, L) - the 3's: input_ids, input_mask, input_key_padding_mask
        """
        # the target sentences, which are to be fed as the inputs to the decoder
        tgt_ids, tgt_key_padding_mask = self.encode([
            # starts with bos, but does not end with eos (pad token is ignored anyways)
            self.tokenizer.bos_token + " " + sent  # noqa
            for sent in tgts
        ])
        tgt_inputs = torch.stack([tgt_ids, tgt_key_padding_mask], dim=1)  # (N, 2, L)
        return tgt_inputs


class InferInputsBuilder(InputsBuilder):

    def __call__(self, srcs: List[str]) -> torch.Tensor:
        src_inputs = self.src_inputs(srcs)  # (N, 2, L)
        tgt_inputs = self.tgt_inputs(len(srcs))  # (N, 2, L)
        inputs = torch.stack([src_inputs, tgt_inputs], dim=1)  # (N, 2, 2, L)
        return inputs

    def tgt_inputs(self, batch_size: int) -> torch.Tensor:
        """
        :param batch_size:
        :return: (N, 2, L)
        """
        tgt_ids, tgt_key_padding_mask = self.encode([
            # just start with bos_token.
            # why no eos token at the end?
            # A: because the label for eos token, i.e. pad token, is ignored in computing the loss anyways
            # also, this may lead to the model repeating characters
            # refer to: https://discuss.pytorch.org/t/transformer-mask-doesnt-do-anything/79765
            self.tokenizer.bos_token  # noqa
            for _ in range(self.max_length)
        ])
        tgt_inputs = torch.stack([tgt_ids, tgt_key_padding_mask], dim=1)  # (N, 2, L)
        return tgt_inputs


class LabelsBuilder(DataBuilder):

    def __call__(self, tgts: List[str]):
        """
        :param tgts:
        :return: (N, L)
        """
        # to be used as the labels
        label_ids, _ = self.encode([
            # does not start with bos, but ends with eos (right-shifted)
            sent + " " + self.tokenizer.eos_token  # noqa
            for sent in tgts
        ])
        return label_ids
