import torch
from torch.utils.data import TensorDataset


def main():
    x = torch.rand(size=(10, 3))
    y = torch.rand(size=(10, 3))
    z = torch.rand(size=(10, 3))

    dataset = TensorDataset(x, y, z)

    print(dataset[0])
    print(dataset[1])
    print(dataset[11])  # should return an error, of course


if __name__ == '__main__':
    main()