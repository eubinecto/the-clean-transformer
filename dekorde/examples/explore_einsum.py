import torch


def main():
    a = torch.Tensor([1, 2, 3])
    b = torch.Tensor([4, 5, 6])
    print(torch.sum(torch.mul(a, b)))
    print(torch.einsum("i,i->", a, b))
    A = torch.Tensor([[1, 2, 3],  # 1 + 2 + 3 = 6
                      [4, 5, 6]])  # 4 + 5 + 6 = 15
    print(torch.einsum('ab->a', A))
    print(torch.sum(A, dim=1))  #
    print(torch.einsum('ab->b', A))
    print(torch.sum(A, dim=0))
    # batched matrix multiplication
    A = torch.randn((10, 20))
    B = torch.randn((20, 30))
    print(A @ B)
    print(torch.einsum("ab,bc->ac", A, B))
    print(torch.einsum("ab,cb->ac", A, B))


if __name__ == '__main__':
    main()
