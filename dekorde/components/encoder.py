import torch
from typing import Tuple
from dekorde.components.mha import MultiHeadAttentionLayer
from dekorde.components.ffn import FeedForward


class EncoderLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, max_length: int, heads: int, dropout: float):
        super().__init__()
        # any layers to optimise?
        self.mhsa_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads, masked=False)
        self.ffn = FeedForward(hidden_size, dropout)
        self.norm_1 = torch.nn.LayerNorm(hidden_size)
        self.norm_2 = torch.nn.LayerNorm(hidden_size)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param inputs: Tuple[src_hidden, src_mask]
        :return: src_hidden: (N, L, H)
        """
        src_hidden, src_mask = inputs
        out_ = self.mhsa_layer.forward(Q=src_hidden, K=src_hidden, V=src_hidden,
                                       key_mask=src_mask)
        out_ = self.norm_1(out_) + src_hidden
        out_ = self.ffn(out_)
        src_hidden = self.norm_2(out_) + out_  # src_hidden is now updated
        return src_hidden, src_mask


class Encoder(torch.nn.Module):
    def __init__(self, hidden_size: int, max_length: int, heads: int, depth: int, dropout: float):
        super().__init__()
        self.encoder_layers = torch.nn.Sequential(
            *[EncoderLayer(hidden_size, max_length, heads, dropout) for _ in range(depth)]
        )

    def forward(self, src_embed: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        :param src_embed: (N, L, H)
        :param src_mask: (N, L)
        :return: input_hidden: (N, L, H)
        """
        input_hidden, _ = self.encoder_layers((src_embed, src_mask))
        return input_hidden
