import torch
import numpy as np
from dekorde.tensors import LookAheadMaskBuilder


class MultiHeadAttentionLayer(torch.nn.Module):
    """
    this could be either masked or not.
    """

    def __init__(self, hidden_size: int, max_length: int, heads: int, masked: bool):
        """
        :param hidden_size:
        :param max_length:
        :param heads:
        :param masked:
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.heads = heads
        self.masked = masked
        # hidden size must be divisible by heads.
        assert hidden_size % heads == 0
        # any layers to optimise? - four linear layers in total.
        self.W_q = torch.nn.Linear(hidden_size, hidden_size)
        self.W_k = torch.nn.Linear(hidden_size, hidden_size)
        self.W_v = torch.nn.Linear(hidden_size, hidden_size)
        self.W_o = torch.nn.Linear(hidden_size, hidden_size)  # for aggregating the multi-head outputs.
        # --- any constant tensors must be registered to a buffer --- #
        self.register_buffer("lookahead_mask", LookAheadMaskBuilder(max_length)())

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        """
        :param Q: (N, L, H)
        :param K: (N, L, H)
        :param V: (N, L, H)
        :param key_mask (N, L)
        :return: hiddens (N, L, H)
        """
        # --- learn patterns from Q, K, V --- #
        Q_ = self.W_q(Q)  # (N, L, H) * (H, H) -> (N, L, H)
        K_ = self.W_k(K)  # (N, L, H) * (H, H) -> (N, L, H)
        V_ = self.W_v(V)  # (N, L, H) * (H, H) -> (N, L, H)
        # transform them into multi-heads
        N = Q_.shape[0]
        # --- split Q, K, V into multi-heads --- #
        # (N, L, H) -> (N, heads, L, H // heads)
        # 각 시간대 (L) 별로, 여러개의 확률분포를 허용한다 (heads).
        # 단, 나중에 모든 해드를 융합했을 때 결국 single head의 출력과 같아지도록,
        # hidden_size = hidden_size / heads 로 설정한다.
        Q_ = Q_.reshape(N, self.heads, self.max_length, self.hidden_size // self.heads)
        K_ = K_.reshape(N, self.heads, self.max_length, self.hidden_size // self.heads)
        V_ = V_.reshape(N, self.heads, self.max_length, self.hidden_size // self.heads)
        # compute the scaled dot product attention
        concats = self.scaled_dot_product_attention(Q_, K_, V_, key_mask)  # ... -> (N, L, H)
        hiddens = self.W_o(concats)  # (N, L, H) * (H, H) -> (N, L, H)
        return hiddens

    def scaled_dot_product_attention(self,
                                     Q: torch.Tensor,
                                     K: torch.Tensor,
                                     V: torch.Tensor,
                                     key_mask: torch.Tensor) -> torch.Tensor:
        """
         # --- einsum symbols --- #
         a = N
         b = heads
         c = H // heads
         i, j = L
        :param Q: (N, heads, L, H // heads)
        :param K: (N, heads, L, H // heads)
        :param V: (N, heads, L, H // heads)
        :param key_mask (N, L)
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
        # the padded tokens are masked
        # (N, L) -> (N, heads, L, L)
        sims = sims.masked_fill(self.build_mask(key_mask) == 0, float("-inf"))
        # then normalise the sims to get the attention scores
        attentions = torch.softmax(sims, dim=3)  # (N, heads, L, L), normalise over keys
        # (N, heads, L, L) * (N, heads, L,  H // heads) -> (N, heads, L, H // heads)
        # contexts_{aicd} = \sum_{j = 1}^{j = L}{attentions_{acij} * V_{ajcd}}
        # that is, we reduce the matrices over the "j" dimension - the key dimension
        contexts = torch.einsum("abij,abjc->abic", attentions, V)
        # heads, H // heads -> H로 reshape하면, 결국엔 concat한 것이랑 같은 결과.
        concats = contexts.reshape(N, self.max_length, self.hidden_size)  # ... -> (N, L, H)
        return concats

    def build_mask(self, key_mask: torch.Tensor):
        """
        :param key_mask: (N, L)
        :return: mask (N,heads, L, L)
        """
        N, L = key_mask.size()
        # (N, L) -> (N, 1, 1, L) -> (N, heads, L, L)
        mask = key_mask.view(N, 1, 1, L)\
                       .expand(-1, self.heads, L, -1)
        # if masked, apply (logical-and it) the lookahead mask
        if self.masked:
            lookahead_mask_ = self.lookahead_mask.view(1, 1, L, L)\
                                  .expand(N, self.heads, -1, -1)
            mask = torch.logical_and(mask, lookahead_mask_)
        return mask
