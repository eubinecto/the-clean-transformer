import torch
from dekorde.components.mha import MultiHeadAttentionLayer
from dekorde.components.ffn import FeedForward


class EncoderLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, max_length: int, heads: int):
        super().__init__()
        # any layers to optimise?
        self.multi_head_self_attention_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads)
        self.norm_1 = torch.nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size)
        self.norm_2 = torch.nn.LayerNorm(hidden_size)

    def forward(self, H_x: torch.Tensor) -> torch.Tensor:
        """
        :param H_x: (N, L, H), or (N, L, E) if this layer is the first layer.
        :return: H_x: (N, L, H)
        """
        Out_ = self.multi_head_self_attention_layer.forward(H_q=H_x, H_k=H_x, H_v=H_x) + H_x
        Out_ = self.norm_1(Out_)
        Out_ = self.ffn(Out_) + Out_
        Out = self.norm_2(Out_)  # this is the new H_x
        return Out  # updated


class Encoder(torch.nn.Module):
    def __init__(self, hidden_size: int, max_length: int, heads: int, depth: int):
        super().__init__()
        self.encoder_layers = torch.nn.Sequential(
            *[EncoderLayer(hidden_size, max_length, heads) for _ in range(depth)]
        )

    def forward(self, X_embed: torch.Tensor) -> torch.Tensor:
        """
        :param X_embed: (N, L, E)
        :return: H_x: (N, L, H)
        """
        return self.encoder_layers(X_embed)
