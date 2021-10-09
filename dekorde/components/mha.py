import numpy as np
import torch


class MultiHeadAttentionLayer(torch.nn.Module):
    """
    this could be either masked or not.
    """
    def __init__(self, hidden_size: int, max_length: int, heads: int, M: torch.Tensor = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.heads = heads
        self.M = M  # attention mask - (L, L)
        # hidden size must be divisible by heads.
        assert hidden_size % heads == 0
        # any layers to optimise? - four linear layers in total.
        self.W_q = torch.nn.Linear(hidden_size, hidden_size)
        self.W_k = torch.nn.Linear(hidden_size, hidden_size)
        self.W_v = torch.nn.Linear(hidden_size, hidden_size)
        self.W_o = torch.nn.Linear(hidden_size, hidden_size)  # for aggregating the multi-head outputs.

    def forward(self, H_q: torch.Tensor, H_k: torch.Tensor, H_v: torch.Tensor) -> torch.Tensor:
        """
        :param H_q: (N, L, H)
        :param H_k: (N, L, H)
        :param H_v: (N, L, H)
        :return: H_all (N, L, H)
        """
        # build Query, Key and Value
        Q = self.W_q(H_q)  # (N, L, H) * (H, H) -> (N, L, H)
        K = self.W_k(H_k)  # (N, L, H) * (H, H) -> (N, L, H)
        V = self.W_v(H_v)  # (N, L, H) * (H, H) -> (N, L, H)
        # transform them into multi-heads
        N = Q.shape[0]
        # (N, L, H) -> (N, heads, L, H // heads)
        Q = Q.reshape(N, self.heads, self.max_length, self.hidden_size // self.heads)
        K = K.reshape(N, self.heads, self.max_length, self.hidden_size // self.heads)
        V = V.reshape(N, self.heads, self.max_length, self.hidden_size // self.heads)
        # compute the scaled dot product attention
        return self.scaled_dot_product_attention(Q, K, V)

    def scaled_dot_product_attention(self,
                                     Q: torch.Tensor,
                                     K: torch.Tensor,
                                     V: torch.Tensor) -> torch.Tensor:
        """
         # --- einsum symbols --- #
         e = heads.
         x = h // heads.
         a = L (first)
         b = L (second
        :param Q: (N, heads, L, H // heads)
        :param K: (N, heads, L, H // heads)
        :param V: (N, heads, L, H // heads)
        :return: H_all (N, L, H)
        """
        N = Q.shape[0]
        # down-scale them to prevent
        Q /= np.sqrt(self.hidden_size)
        K /= np.sqrt(self.hidden_size)
        Sims = torch.einsum("neax,nebx->neab", Q, K)  # ... -> (N, heads, L, L)
        if self.M is not None:  # masked self attention
            Sims[self.M.expand(N, self.heads, self.max_length, self.max_length)] = float("-inf")
        Attentions = torch.softmax(Sims, dim=2)  # (N, heads, L, L)
        # (N, heads, L, L) * (N, heads, L,  H // heads) -> (N, heads, L,  H // heads)
        Contexts = torch.einsum("neab,neax->neax", Attentions, V)
        Concats = Contexts.reshape(N, self.max_length, self.hidden_size)  # ... -> (N, L, H)
        H_all = self.W_o(Concats)  # (N, L, H) * (H, H) -> (N, L, H)
        return H_all
