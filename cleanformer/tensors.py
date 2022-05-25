"""
any functions for building tensors are defined here
"""
from typing import List, Tuple
import torch
import numpy as np
from tokenizers import Encoding, Tokenizer


# --- data-related --- #
def src(tokenizer: Tokenizer, max_length: int, sents: List[str]) -> torch.Tensor:
    """
    returns: src (N, 2, L)
    """
    # the source sentences, which are to be fed as the inputs to the encoder
    src_ids, src_key_padding_mask = tokens(tokenizer, max_length, [
        # source sentence does not need a bos token, because, unlike decoder,
        # what encoder does is simply extracting  contexts from src_ids
        # https://github.com/Kyubyong/transformer/issues/64
        sent + " " + tokenizer.eos_token  # noqa
        for sent in sents
    ])
    return torch.stack([src_ids, src_key_padding_mask], dim=1).long()  # (N, 2, L)


def tgt_r(tokenizer: Tokenizer, max_length: int, sents: List[str]) -> torch.Tensor:
    # the target sentences, which are to be fed as the inputs to the decoder
    tgt_r_ids, tgt_r_key_padding_mask = tokens(tokenizer, max_length, [
        # starts with bos, but does not end with eos (pad token is ignored anyways)
        tokenizer.bos_token + " " + sent  # noqa
        for sent in sents
    ])
    return torch.stack([tgt_r_ids, tgt_r_key_padding_mask], dim=1).long()  # (N, 2, L)


def tgt(tokenizer: Tokenizer, max_length: int, sents: List[str]) -> torch.Tensor:
    tgt_ids, _ = tokens(tokenizer, max_length, [
        # does not start with bos, but ends with eos (right-shifted)
        sent + " " + tokenizer.eos_token  # noqa
        for sent in sents
    ])
    return tgt_ids  # (N, L)

def subsequent_mask(max_length: int) -> torch.LongTensor:
    """
    a square matrix for auto-regressively (i.e. subsequently) allowing positions
    return: (L, L)
    """
    mask = torch.tril(torch.ones(size=(max_length, max_length)), diagonal=0).long()  # (L, L) -> (L, L)
    return mask


def pos_encodings(max_length: int, hidden_size: int) -> torch.Tensor:
    """
    :return: (L, H)
    """
    positions = torch.arange(max_length).view(-1, 1)  # (,) -> (L)
    freqs = 0.0001**(torch.arange(hidden_size)[::2] / hidden_size).view(1, -1)  # (,) -> (H)
    encodings = torch.zeros(size=(max_length, hidden_size))  # (L, H)
    # fill in the pairs by broadcast-multiplying freqs to positions
    encodings[:, ::2] = torch.sin(freqs * positions)   # evens = sin
    # odds = cos, but with the same frequency as above
    # why the same frequency?
    # A: so that dist(PE(pos + k) - PE(pos)) stays constant
    encodings[:, 1::2] = torch.cos(freqs * positions)
    return encodings


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 key_mask: torch.LongTensor) -> torch.Tensor:
    """
    Definition: Attention(Q, K, V) = softmax(Q dot K^T / sqrt(d_k)) dot V
    What it does: soft-align values with respect to the similarities of their keys to each query
    param q: (..., L, H)
    param k: (..., L, H)
    param v: (..., L, H)
    key_mask: (..., L, L)
    """
    # Q * K^T:  compute query-key similarities
    sims = torch.einsum("...qh,...kh->...qk", q, k)
    # Q * K^T / sqrt(d_k): down-scale similarities to prevent gradient vanishing
    sims /= np.sqrt(k.shape[-1])
    # apply padding mask and/or subsequent mask
    sims = sims.masked_fill(key_mask == 0, float("-inf"))
    # softmax(Q * K^T / sqrt(d_k)): normalise the sims over keys
    attentions = torch.softmax(sims, dim=-1)
    # softmax(Q * K^T / sqrt(d_k)) * V: soft-align values with respect to each query
    alignments = torch.einsum("...qv,...vh->...qh", attentions, v)
    return alignments


def tokens(tokenizer: Tokenizer, max_length: int, sents: List[str]) -> Tuple[torch.LongTensor, torch.LongTensor]:
    # we use enable padding here, rather than on creating the tokenizer,
    # to set maximum length on encoding
    tokenizer.enable_padding(pad_token=tokenizer.pad_token,  # noqa
                             pad_id=tokenizer.pad_token_id,  # noqa
                             length=max_length)
    # don't add special tokens, we will add them ourselves
    encodings: List[Encoding] = tokenizer.encode_batch(sents, add_special_tokens=False)
    ids = torch.LongTensor([encoding.ids for encoding in encodings])
    key_padding_mask = torch.LongTensor([encoding.attention_mask for encoding in encodings])
    return ids, key_padding_mask
