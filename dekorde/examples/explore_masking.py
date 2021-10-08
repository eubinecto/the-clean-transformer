
from dekorde.builders import build_M
from dekorde.loaders import load_device
import torch


def main():
    device = load_device()
    L = 10
    M = build_M(L, device)
    print(M)
    N = 30
    heads = 8
    H = 512
    # multi-head scores.
    S = torch.randn(size=(N, heads, L, L))
    # 아하, 간단하네.
    S[M.expand(N, heads, L, L) == 0] = float("-inf")
    print(S)

if __name__ == '__main__':
    main()