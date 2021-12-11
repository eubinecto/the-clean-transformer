"""
any constant tensors are defined here.
they will be registered as buffers.
"""
import torch


def no_mask(max_length: int) -> torch.LongTensor:
    """
    Just allow all positions
    """
    mask = torch.zeros(size=(max_length, max_length)).long()
    return mask


def subsequent_mask(max_length: int) -> torch.LongTensor:
    """
    Subsequently allow positions
    """
    mask = torch.triu(torch.ones(size=(max_length, max_length)), diagonal=1).long()  # (L, L) -> (L, L)
    return mask


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
