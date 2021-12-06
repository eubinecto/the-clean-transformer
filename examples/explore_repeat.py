import torch


def main():
    N = 30
    L = 50
    positions = torch.arange(L)
    print(positions.repeat(N, 1).shape)


if __name__ == '__main__':
    main()