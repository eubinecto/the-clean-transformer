import math

import torch
import torch.nn.functional as F


class MultiHeadAttentionLayer(torch.nn.Module):
    """
    this could be either masked or not.
    """
    def __init__(self, embed_size: int, hidden_size: int, heads: int):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.heads = heads
        # any layers to optimise? - four linear layers in total.
        # TODO - define the shape of the weights.
        self.W_q = torch.nn.Linear(..., ...)
        self.W_k = torch.nn.Linear(..., ...)
        self.W_v = torch.nn.Linear(..., ...)
        self.W_o = torch.nn.Linear(..., ...)  # for aggregating the multi-head outputs.

    def forward(self, EP_q: torch.Tensor, EP_k: torch.Tensor, EP_v: torch.Tensor, M: torch.Tensor = None) -> torch.Tensor:
        """
        :param EP_q: (N, L, E)
        :param EP_k: (N, L, E)
        :param EP_v: (N, L, E)
        :param M: (???) The mask.
        :return: H_all (N, L, H)
        """
        Q = self.W_q(EP_q)
        K = self.W_k(EP_k)
        V = self.W_v(EP_v)
        return self.scaled_dot_product_attention(Q, K, V, M)

    @staticmethod
    def scaled_dot_product_attention(Q: torch.Tensor,
                                     K: torch.Tensor,
                                     V: torch.Tensor,
                                     M: torch.Tensor = None) -> torch.Tensor:
        """
        :param Q: (N, L, H)
        :param K: (N, L, H)
        :param V: (N, L, H)
        :param M: (???)
        :return: H_all (N, L, H)
        """
        d_k = Q.size(-1)
        scores: torch.tensor = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)        # 수식에 적혀있는 그대로 구현.
        if M:  # if not None.
            scores = scores.masked_fill(M == 0, -1e9)        # M == 0 의 조건을 충족한다면 scores 에 해당 위치의 값을 -1e9로 대체.
        prob_attn = F.softmax(scores, dim=-1)               # 어텐션 확률

        raise torch.matmul(prob_attn, V)

