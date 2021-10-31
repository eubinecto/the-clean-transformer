from typing import Tuple

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

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param inputs: Tuple[H_x, padding_mask]
        :return: H_x: (N, L, H)
        """
        H_x, padding_mask = inputs
        Out_ = self.multi_head_self_attention_layer.forward(H_q=H_x, H_k=H_x, H_v=H_x, mask=padding_mask) + H_x
        Out_ = self.norm_1(Out_)
        Out_ = self.ffn(Out_) + Out_
        Out = self.norm_2(Out_)  # this is the new H_x
        return Out, padding_mask


class Encoder(torch.nn.Module):
    def __init__(self, hidden_size: int, max_length: int, heads: int, depth: int):
        super().__init__()
        self.encoder_layers = torch.nn.Sequential(
            *[EncoderLayer(hidden_size, max_length, heads) for _ in range(depth)]
        )

    def forward(self, X_embed: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        :param X_embed: (N, L, E)
        :param padding_mask: (N, L)
        :return: H_x: (N, L, H)
        """
        H_x, _ = self.encoder_layers((X_embed, padding_mask))
        return H_x
