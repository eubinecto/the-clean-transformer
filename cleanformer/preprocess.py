from typing import List, Tuple
import torch  # noqa
from tokenizers import Encoding, Tokenizer  # noqa


def encode(
    tokenizer: Tokenizer, max_length: int, sents: List[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    tokenize & encoder
    """
    # we use enable padding here, rather than on creating the tokenizer,
    # to set maximum length on encoding
    tokenizer.enable_padding(
        pad_token=tokenizer.pad_token,  # noqa
        pad_id=tokenizer.pad_token_id,  # noqa
        length=max_length,
    )
    # don't add special tokens, we will add them ourselves
    encodings: List[Encoding] = tokenizer.encode_batch(sents, add_special_tokens=False)
    ids = torch.LongTensor([encoding.ids for encoding in encodings])
    key_padding_mask = torch.LongTensor(
        [encoding.attention_mask for encoding in encodings]
    )
    return ids, key_padding_mask


def src(
    tokenizer: Tokenizer, max_length: int, x2y: List[Tuple[str, str]]
) -> torch.Tensor:
    """
    returns: src (N, 2, L)
    """
    sents = [x for x, _ in x2y]
    # the source sentences, which are to be fed as the inputs to the encoder
    src_ids, src_key_padding_mask = encode(
        tokenizer,
        max_length,
        [
            # source sentence does not need a bos token, because, unlike decoder,
            # what encoder does is simply extracting  contexts from src_ids
            # https://github.com/Kyubyong/transformer/issues/64
            sent + " " + tokenizer.eos_token  # noqa
            for sent in sents
        ],
    )
    return torch.stack([src_ids, src_key_padding_mask], dim=1).long()  # (N, 2, L)


def tgt_r(
    tokenizer: Tokenizer, max_length: int, x2y: List[Tuple[str, str]]
) -> torch.Tensor:
    """
    return: tgt_r (N, 2, L)
    """
    sents = [y for _, y in x2y]
    # the target sentences, which are to be fed as the inputs to the decoder
    tgt_r_ids, tgt_r_key_padding_mask = encode(
        tokenizer,
        max_length,
        [
            # starts with bos, but does not end with eos (pad token is ignored anyways)
            tokenizer.bos_token + " " + sent  # noqa
            for sent in sents
        ],
    )
    return torch.stack([tgt_r_ids, tgt_r_key_padding_mask], dim=1).long()  # (N, 2, L)


def tgt(
    tokenizer: Tokenizer, max_length: int, x2y: List[Tuple[str, str]]
) -> torch.Tensor:
    sents = [y for _, y in x2y]
    tgt_ids, _ = encode(
        tokenizer,
        max_length,
        [
            # does not start with bos, but ends with eos (right-shifted)
            sent + " " + tokenizer.eos_token  # noqa
            for sent in sents
        ],
    )
    return tgt_ids  # (N, L)
