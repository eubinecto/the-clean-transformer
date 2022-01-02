"""
any functions for building tensors are defined here
"""
import torch
import numpy as np


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
