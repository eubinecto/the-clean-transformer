
from cleanformer.tensors import subsequent_mask
import torch


def main():
    N = 30
    L = 10
    heads = 8
    mask = subsequent_mask(L)
    # multi-head scores.
    S = torch.randn(size=(N, heads, L, L))
    # 아하, 간단하네.
    S = S.masked_fill(mask == 1, float("-inf"))
    print(S)


if __name__ == '__main__':
    main()
