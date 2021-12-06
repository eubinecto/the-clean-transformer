

import torch


def main():
    # this is.. probably useful for building the attention mask
    N = 10
    I = torch.ones(size=(N, N))
    M = torch.tril(I,
                   # if we set this to 0, then it will start from .
                   diagonal=0)
    print(M)


if __name__ == '__main__':
    main()
