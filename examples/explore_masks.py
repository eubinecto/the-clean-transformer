import torch


def main():
    N = 5
    heads = 7
    L = 15
    # build a query mask
    query_mask = torch.randint(low=0, high=2, size=(N, L))\
                      .view((N, 1, L, 1))\
                      .expand(-1, heads, -1, L)
    print(query_mask.shape)
    print(query_mask[0][0])

    # build a key mask
    key_mask = torch.randint(low=0, high=2, size=(N, L))\
                    .view((N, 1, 1, L))\
                    .expand(-1, heads, L, -1)
    print(key_mask.shape)
    print(key_mask[0][0])

    # the final mask
    mask = torch.logical_and(query_mask, key_mask)
    print(mask[0][0])

if __name__ == '__main__':
    main()
