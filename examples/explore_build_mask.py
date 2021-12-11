import torch

from enkorde.tensors import subsequent_mask


def main():
    heads = 3
    # here is an example
    key_mask = torch.LongTensor([[1, 1, 1, 0, 0],
                                 [1, 1, 1, 1, 0]])
    N, L = key_mask.size()
    # (N, L) -> (N, 1, 1, L) -> (N, heads, L, L)
    mask = key_mask.view(N, 1, 1, L) \
                   .repeat(1, heads, L, 1)
    print("### Only the padding masks ###")
    print(mask)
    print(mask.shape)

    # (N, L) -> (N, 1, 1, L) -> (N, heads, L, L)
    sub_mask = subsequent_mask(L).view(1, 1, L, L) \
                                 .repeat(N, heads, 1, 1)
    # (N, heads, L, L), (N, heads, L, L) -> (N, heads, L, L)
    mask = torch.logical_and(mask, sub_mask).long()
    print("### With subsequent mask ###")
    print(mask)
    print(mask.shape)


if __name__ == '__main__':
    main()
