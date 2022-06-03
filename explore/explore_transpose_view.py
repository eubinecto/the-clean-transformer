"""
다음 참고:
https://sanghyu.tistory.com/3
"""
import torch


def main():
    X = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    N, L = X.size()
    print(X)
    print(X.view(L, N))  # contiguous
    print(X.transpose(0, 1))  # non-contiguous


if __name__ == "__main__":
    main()
