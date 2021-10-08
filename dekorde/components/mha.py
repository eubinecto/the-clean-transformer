import torch
import torch.nn as nn


# FIXME 변경한 점: Mask Multi Head Attention Layer를 따로 구현함
class MultiHeadAttentionLayer(nn.Module):
    """
    this could be either masked or not.
    """
    # 의문: 여기서의 hidden_size가 query, key, vector의 size를 말하는 것인지?
    # hidden_size의 역할이 무엇이었을지 모르겠습니당
    def __init__(self, embed_size: int, heads: int, masked: bool):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = embed_size // heads
        self.heads = heads
        self.masked = masked

        # any layers to optimise? - four linear layers in total.
        self.W_q = torch.nn.Linear(embed_size, self.hidden_size * heads) # head 개 만큼의 hidden_size 차원을 갖는 query vector를 만든다
        self.W_k = torch.nn.Linear(embed_size, self.hidden_size * heads)
        self.W_v = torch.nn.Linear(embed_size, self.hidden_size * heads)
        self.W_o = torch.nn.Linear(self.hidden_size * heads, embed_size)  # for aggregating the multi-head outputs.

    # TODO: EP의 뜻이 뭔지...??
    def forward(self, EP_q: torch.Tensor, EP_k: torch.Tensor, EP_v: torch.Tensor, Mask: torch.Tensor = None) -> torch.Tensor:
        """
        :param EP_q: (N, L, E)
        :param EP_k: (N, L, E)
        :param EP_v: (N, L, E)
        :param M: (???) The mask.
        :return: H_all (N, L, H)
        """
        Q = self.W_q(EP_q) # (B, L, H * N_head)
        K = self.W_k(EP_k) # (B, L, H * N_head)
        V = self.W_v(EP_v) # (B, L, H * N_head)
        return self.scaled_dot_product_attention(Q, K, V, Mask)

    def scaled_dot_product_attention(self,
                                     Q: torch.Tensor,
                                     K: torch.Tensor,
                                     V: torch.Tensor, 
                                     Mask: torch.Tensor) -> torch.Tensor:
        """
        :param Q: (N, L, H)
        :param K: (N, L, H)
        :param V: (N, L, H)
        :param M: (???)
        :return: H_all (N, L, H)
        """
        attn_scores = torch.matmul(Q, K) # (N, L)
        
        # Masked Multi Head Attention
        if self.masked: 
            assert ( Mask != None ), "Masked Multi head attention require mask as input"
            attn_scores = attn_scores.masked_fill(Mask == 0, -float('inf'))
        V = torch.dot(attn_scores, V) # (N, L, H)
        return V
