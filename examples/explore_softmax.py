import torch


def main():
    X = torch.Tensor([
        [1, 2, 3],
        [4, 5, 6]
    ])  # (N, L)
    print(torch.softmax(X, dim=0))  # normalise over N
    print(torch.softmax(X, dim=1))  # normalise over L



if __name__ == '__main__':
    main()