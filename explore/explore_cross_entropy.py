
from torch.nn import functional as F
import torch


def main():
    Y_pred = torch.FloatTensor([
        [10000, float("-inf"), float("-inf")],
        [float("-inf"), 10000, float("-inf")]
    ])
    Y = torch.LongTensor([0, 1])
    # should be close to zero.
    print(F.cross_entropy(Y_pred, Y))


if __name__ == '__main__':
    main()
