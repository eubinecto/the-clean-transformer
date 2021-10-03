
import torch


class MultiHeadAttentionLayer(torch.nn.Module):
    """
    this could be either masked or not.
    """
    def __init__(self, heads: int):
        self.heads = heads
        super().__init__()

    def forward(self) -> torch.Tensor:
        pass

    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                     masked: bool) -> torch.Tensor:
        pass

