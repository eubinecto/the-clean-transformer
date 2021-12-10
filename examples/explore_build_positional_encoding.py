
import torch
import numpy as np


def main():
    max_length = 10
    hidden_size = 32
    positions = torch.arange(max_length)  # (,) -> (L)
    print(positions.shape)
    print(positions[::2])  # only the even numbers
    print(positions[1::2])  # only the odd numbers
    freqs = 0.0001**(2*(torch.arange(hidden_size))/hidden_size)  # (,) -> (H)
    print(freqs.shape)
    print(positions.unsqueeze(1))
    print(freqs.unsqueeze(1))
    # broadcast multiplication (broadcast positions to each frequency)
    encodings = positions.unsqueeze(1) * freqs.unsqueeze(0)
    print(encodings.shape)
    print(encodings[:, ::2])
    # fill in the pairs
    # evens = cos
    # odds = sin
    encodings[:, ::2] = np.cos(encodings[:, ::2])
    encodings[:, 1::2] = np.sin(encodings[:, ::2])


if __name__ == '__main__':
    main()
