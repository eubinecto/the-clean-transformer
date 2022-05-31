import torch  # noqa
from cleanformer.models import functional as cF  # noqa


class MultiHeadAttentionLayer(torch.nn.Module):
    """
    this could be either masked or not.
    """

    def __init__(self, hidden_size: int, max_length: int, heads: int, masked: bool):
        """
        :param hidden_size:
        :param max_length:
        :param heads: the number of heads
        :param masked: set this to True if you want to apply subsequent mask as well as padding mask to
        a query-key similarity matrix, False if you want to apply only the padding mask to the matrix
        """
        super().__init__()
        assert hidden_size % heads == 0, "hidden size is not divisible by heads"
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.heads = heads
        self.masked = masked
        self.head_size = hidden_size // heads
        # --- layers to optimise --- #
        self.linear_q = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_k = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_v = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_o = torch.nn.Linear(hidden_size, hidden_size)
        self.norm = torch.nn.LayerNorm(hidden_size)
        # --- constant tensors --- #
        self.register_buffer("subsequent_mask", cF.subsequent_mask(max_length))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.LongTensor,
    ) -> torch.Tensor:
        """
        :param q: (N, L, H)
        :param k: (N, L, H)
        :param v: (N, L, H)
        :param key_padding_mask (N, L)
        :return: alignments (N, L, H)
        """
        N = q.shape[0]
        # linear transformation of q, k and v
        q = self.linear_q(q)  # (N, L, H) * (H, H) -> (N, L, H)
        k = self.linear_k(k)  # (N, L, H) * (H, H) -> (N, L, H)
        v = self.linear_v(v)  # (N, L, H) * (H, H) -> (N, L, H)
        # split q, k, v into multi-heads
        q = q.view(
            N, self.max_length, self.heads, self.head_size
        )  # (N, L, H) -> (N, L, heads, head_size)
        k = k.view(
            N, self.max_length, self.heads, self.head_size
        )  # (N, L, H) -> (N, L, heads, head_size)
        v = v.view(
            N, self.max_length, self.heads, self.head_size
        )  # (N, L, H) -> (N, L, heads, head_size)
        # make q, k and v matmul-compatible
        q = q.transpose(1, 2)  # (N, L, heads, head_size) -> (N, heads, L, head_size)
        k = k.transpose(1, 2)  # (N, L, heads, head_size) -> (N, heads, L, head_size)
        v = v.transpose(1, 2)  # (N, L, heads, head_size) -> (N, heads, L, head_size)
        # key mask = key padding mask: ignore [PAD] tokens
        key_mask = key_padding_mask.view(N, 1, 1, self.max_length).expand(
            -1, self.heads, self.max_length, -1
        )  # (N, L) -> (N, 1, 1, L) -> (N, heads, L, L)
        # if masked, key mask = key padding mask && key subsequent mask: ignore subsequent positions as well
        if self.masked:
            key_subsequent_mask = self.subsequent_mask.view(
                1, 1, self.max_length, self.max_length
            ).expand(
                N, self.heads, -1, -1
            )  # (L, L) -> (1, 1, L, L) -> (N, heads, L, L)
            key_mask = torch.logical_and(key_mask, key_subsequent_mask).long()
        # soft-align values with respect to the similarities of their keys to each query
        alignments = cF.scaled_dot_product_attention(q, k, v, key_mask)
        # cat(head_1, head_2, ... head_heads): concatenate multiple alignments
        # (N, heads, L, head_size) -> (N, L, heads, head_size) -> (N, L, H)
        cats = (
            alignments.transpose(1, 2)
            .contiguous()
            .view(-1, self.max_length, self.hidden_size)
        )
        # cat(head_1, head_2, ... head_heads) * W_o: aggregate alignments
        alignments = self.linear_o(cats)  # (N, L, H) * (H, H) -> (N, L, H)
        return self.norm(alignments)  # layer normalisation
