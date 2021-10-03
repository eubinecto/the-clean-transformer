import torch


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

    def scaled_dot_product_attention(self,
                                     Q: torch.Tensor,
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
        # TODO
        ...
        if M:  # if not None.
            ...
        ...
        raise NotImplementedError

