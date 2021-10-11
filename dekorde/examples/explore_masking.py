
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
    S[lookahead_mask.expand(N, heads, L, L) == 0] = float("-inf")
    print(S)


if __name__ == '__main__':
    main()
