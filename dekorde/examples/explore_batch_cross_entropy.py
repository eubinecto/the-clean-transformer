
from torch.nn import functional as F
import torch


def main():
    N = 10
    L = 30
    V = 100
    Y_pred = torch.randn(size=(N, L, V))
    Y_pred = torch.softmax(Y_pred, dim=2)
    Y = torch.randint(low=0, high=V, size=(N, L))
    print(Y[0])
    loss = F.cross_entropy(Y_pred, Y, ignore_index=1)


if __name__ == '__main__':
    main()
