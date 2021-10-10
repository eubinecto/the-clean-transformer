import torch

from dekorde.components.mha import MultiHeadAttentionLayer

Q = torch.tensor(
    [[1, 2, 3],
     [11, 22, 33]]
)
K = torch.tensor(
    [[4, 5, 6],
     [44, 55, 66]]
)
V = torch.tensor(
    [[0.15, 0.25, 0.35],
     [0.151, 0.252, 0.353]]
)


def explore_scaled_dot_prod_attn_with_M():
    M = torch.tensor(
        [[0, 1],
         [1, 1]]
    )

    res = MultiHeadAttentionLayer.scaled_dot_product_attention(Q, K, V, M)
    print(res)


def explore_scaled_dot_prod_attn_wo_M():
    M = None

    res = MultiHeadAttentionLayer.scaled_dot_product_attention(Q, K, V, M)
    print(res)


if __name__ == '__main__':
    explore_scaled_dot_prod_attn_with_M()
    explore_scaled_dot_prod_attn_wo_M()
