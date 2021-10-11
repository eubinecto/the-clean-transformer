
from dekorde.builders import build_lookahead_mask
from dekorde.loaders import load_device
import torch


def main():
    N = 30
    L = 10
    heads = 8
    device = load_device()
    lookahead_mask = build_lookahead_mask(L, device)
    # multi-head scores.
    S = torch.randn(size=(N, heads, L, L))
    # 아하, 간단하네.
    S = torch.masked_fill(S, lookahead_mask == 0, value=float("-inf"))
    print(S)


if __name__ == '__main__':
    main()
