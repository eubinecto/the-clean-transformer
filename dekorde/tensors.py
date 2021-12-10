"""
any constant tensors are defined here.
they will be registered as buffers.
"""
import torch


def subsequent_mask(max_length: int) -> torch.LongTensor:
    """
    :return: (L, L)
    """
    ones = torch.ones(size=(max_length, max_length))  # ... -> (L, L)
    return torch.tril(ones, diagonal=0).long()  # (L, L) -> (L, L)


def pos_encodings(max_length: int, hidden_size: int) -> torch.Tensor:
    """
    :return: (L, H)
    """
    positions = torch.arange(max_length).view(-1, 1)  # (,) -> (L)
    freqs = 0.0001**(torch.arange(hidden_size)[::2] / hidden_size).view(1, -1)  # (,) -> (H)
    # broadcast multiplication (broadcast positions to each frequency)
    encodings = torch.zeros(size=(max_length, hidden_size))  # (L, H)
    # fill in the pairs
    encodings[:, ::2] = torch.sin(freqs * positions)   # evens = sin
    encodings[:, 1::2] = torch.cos(freqs * positions)  # odds = cos, with the same frequency as above
    # hence, PE(pos + k) - PE(pos) stays constant
    return encodings
