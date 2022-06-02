import torch


def main():
    x = torch.rand((10, 3))
    print(x)
    # should work?
    print(x.numpy())


if __name__ == "__main__":
    main()
