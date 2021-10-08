from typing import Tuple
import torch
from dekorde.components.mha import MultiHeadAttentionLayer
from dekorde.components.ffn import FeedForward


class DecoderLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, max_length: int, heads: int, M: torch.Tensor):
        super().__init__()
        # masked
        self.masked_multi_head_self_attention_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads, M)
        self.norm_1 = torch.nn.LayerNorm(self.hidden_size)
        # not masked
        self.multi_head_encoder_decoder_attention_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads)
        self.norm_2 = torch.nn.LayerNorm(self.hidden_size)
        self.ffn = FeedForward(hidden_size)
        self.norm_3 = torch.nn.LayerNorm(self.hidden_size)

    def forward(self, H_x: torch.Tensor, H_y: torch.Tensor, attention_mask: torch.Tensor)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param H_x: (N, L, H)
        :param H_y: (N, L, H)
        :param attention_mask: (L,)
        :return: H_x (as-is), H_y (updated), attention_mask (as-is)
        """
        Out_ = self.masked_multi_head_self_attention_layer\
                   .forward(H_q=H_y, H_k=H_y, H_v=H_y) + H_y
        Out_ = self.norm_1(Out_)
        Out_ = self.multi_head_encoder_decoder_attention_layer.forward(H_q=Out_, H_k=H_x, H_v=H_x) + Out_
        Out_ = self.norm_2(Out_)
        Out_ = self.ffn(Out_)
        Out = self.norm_3(Out_)  # H_y updated
        return H_x, Out, attention_mask


class Decoder(torch.nn.Module):

    def __init__(self, hidden_size: int, max_length: int, heads: int, depth: int, M: torch.Tensor):
        super().__init__()
        self.decoder_layers = torch.nn.Sequential(
            *[DecoderLayer(hidden_size, max_length, heads, M) for _ in range(depth)]
        )

    def forward(self, H_x: torch.Tensor, Y_embed: torch.Tensor) -> torch.Tensor:
        """
        :param H_x: (N, L, H)
        :param Y_embed: (N, L, H)
        :return: H_y: (N, L, H)
        """
        return self.decoder_layers(H_x, Y_embed)
