import torch


def explore_expand():
    a = torch.tensor([[1, 2, 3], [4, 5, 6]])

    print(a.expand(1, 1, 2))


if __name__ == '__main__':
    explore_expand()
