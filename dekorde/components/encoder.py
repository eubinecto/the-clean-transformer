import torch
from dekorde.components.mha import MultiHeadAttentionLayer


class EncoderLayer(torch.nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, heads: int):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.heads = heads
        # any layers to optimise?
        self.multi_head_self_attention_layer = MultiHeadAttentionLayer(heads, masked=False)
        self.norm_1 = torch.nn.LayerNorm(...)
        self.ffn = torch.nn.Linear(..., ...)
        self.norm_2 = torch.nn.LayerNorm(...)

    def forward(self, X_ep: torch.Tensor) -> torch.Tensor:
        """
        :param X_ep: (N, L, E)
        :return: H_all_x: (N, L, H)
        """
        # TODO - residual connection, layer norm.
        raise NotImplementedError


class Encoder(torch.nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, heads:int, depth: int):
        super().__init__()
        self.encoder_layers = torch.nn.Sequential(
            *[EncoderLayer(embed_size, hidden_size, heads) for _ in range(depth)]
        )

    def forward(self, X_ep: torch.Tensor) -> torch.Tensor:
        """
        :param X_ep: (N, L, E)
        :return: H_all_x: (N, L, H)
        """
        return self.encoder_layers(X_ep)
