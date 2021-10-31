import numpy as np
import torch


class MultiHeadAttentionLayer(torch.nn.Module):
    """
    this could be either masked or not.
    """

    def __init__(self, hidden_size: int, max_length: int, heads: int):
        """
        :param hidden_size:
        :param max_length:
        :param heads:
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.heads = heads
        # hidden size must be divisible by heads.
        assert hidden_size % heads == 0
        # any layers to optimise? - four linear layers in total.
        self.W_q = torch.nn.Linear(hidden_size, hidden_size)
        self.W_k = torch.nn.Linear(hidden_size, hidden_size)
        self.W_v = torch.nn.Linear(hidden_size, hidden_size)
        self.W_o = torch.nn.Linear(hidden_size, hidden_size)  # for aggregating the multi-head outputs.

    def forward(self, H_q: torch.Tensor, H_k: torch.Tensor, H_v: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        :param H_q: (N, L, H)
        :param H_k: (N, L, H)
        :param H_v: (N, L, H)
        :param mask (N, L)
        :return: H_all (N, L, H)
        """
        # build Query, Key and Value
        Q = self.W_q(H_q)  # (N, L, H) * (H, H) -> (N, L, H)
        K = self.W_k(H_k)  # (N, L, H) * (H, H) -> (N, L, H)
        V = self.W_v(H_v)  # (N, L, H) * (H, H) -> (N, L, H)
        # transform them into multi-heads
        N = Q.shape[0]
        # (N, L, H) -> (N, heads, L, H // heads)
        # 각 시간대 (L) 별로, 여러개의 확률분포를 허용한다 (heads).
        # 단, 나중에 모든 해드를 융합했을 때 결국 single head의 출력과 같아지도록,
        # hidden_size = hidden_size / heads 로 설정한다.
        Q = Q.reshape(N, self.heads, self.max_length, self.hidden_size // self.heads)
        K = K.reshape(N, self.heads, self.max_length, self.hidden_size // self.heads)
        V = V.reshape(N, self.heads, self.max_length, self.hidden_size // self.heads)
        # compute the scaled dot product attention
        concats = self.scaled_dot_product_attention(Q, K, V, mask)  # ... -> (N, L, H)
        H_all = self.W_o(concats)  # (N, L, H) * (H, H) -> (N, L, H)
        return H_all

    def scaled_dot_product_attention(self,
                                     Q: torch.Tensor,
                                     K: torch.Tensor,
                                     V: torch.Tensor,
                                     mask: torch.Tensor) -> torch.Tensor:
        """
         # --- einsum symbols --- #
         a = N
         b = heads
         c = H // heads
         i, j = L
        :param Q: (N, heads, L, H // heads)
        :param K: (N, heads, L, H // heads)
        :param V: (N, heads, L, H // heads)
        :param mask (N, L)
        :return: concats (N, L, H)
        """
        N = Q.shape[0]
        # 행렬곱 전에 미리 scale.
        # 행렬곱 이후에 스케일하면 소 잃고 외양간 고치는 격.
        Q /= np.sqrt(self.hidden_size)
        K /= np.sqrt(self.hidden_size)
        # (N, heads, L, H // heads) * (N, heads, L, H // heads) -> (N, heads, L, L)
        # sims_{abij} = \sum_{d = 1}^{d= H // heads}{Q_{abic} * K_{abjc}}
        # that is, we reduce the matrices over the "d" dimension
        sims = torch.einsum("abic,abjc->abij", Q, K)
        # mask the sims with the padding mask
        sims = self.mask_sims(sims, mask=mask)
        # then normalise the sims to get the attention scores
        attentions = torch.softmax(sims, dim=2)  # (N, heads, L, L), normalise over L (the first one)
        # (N, heads, L, L) * (N, heads, L,  H // heads) -> (N, heads, L, H // heads)
        # contexts_{aicd} = \sum_{j = 1}^{j = L}{attentions_{acij} * V_{ajcd}}
        # that is, we reduce the matrices over the "j" dimension
        contexts = torch.einsum("abij,abjc->abic", attentions, V)
        # heads, H // heads -> H로 reshape하면, 결국엔 concat한 것이랑 같은 결과.
        concats = contexts.reshape(N, self.max_length, self.hidden_size)  # ... -> (N, L, H)
        return concats

    def mask_sims(self, sims: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param sims: (N, heads, L, L)
        :param mask: (N, L) if padding_mask, or  (L, L) if lookahead mask
        :return:
        """

        sims = sims.masked_fill(
            mask.repeat(
                list(sims.shape[:2])+[sims.shape[2] // mask.shape[0], sims.shape[3] // mask.shape[1]]
            ) == 0,
            float("-inf")
        )
        return sims


class MaskedMultiHeadAttentionLayer(MultiHeadAttentionLayer):
    def __init__(self, hidden_size: int, max_length: int, heads: int, lookahead_mask: torch.Tensor):
        super().__init__(hidden_size, max_length, heads)
        self.lookahead_mask = lookahead_mask

    def mask_sims(self, sims: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # then mask the answers
        sims = super(MaskedMultiHeadAttentionLayer, self).mask_sims(sims, mask=self.lookahead_mask)
        return sims
