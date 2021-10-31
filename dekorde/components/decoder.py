from typing import Tuple
import torch
from dekorde.components.mha import MultiHeadAttentionLayer
from dekorde.components.ffn import FeedForward


class DecoderLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, max_length: int, heads: int):
        super().__init__()
        # masked, multi-head self-attention layer.
        self.masked_mhsa_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads)
        self.norm_1 = torch.nn.LayerNorm(hidden_size)
        # not masked, multi-head encoder-decoder attention layer.
        self.mheda_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads)
        self.norm_2 = torch.nn.LayerNorm(hidden_size)
        # position-wise feed fowrard network.
        self.ffn = FeedForward(hidden_size)
        self.norm_3 = torch.nn.LayerNorm(hidden_size)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor])\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param inputs: (H_x = (N, L, H), H_y = (N, L, H), padding_mask (N, L))
        :return: H_x (as-is), H_y (updated), padding_mask (as-is)
        """
        H_x, H_y, padding_mask, lookahead_mask = inputs
        Out_ = self.masked_mhsa_layer.forward(H_q=H_y, H_k=H_y, H_v=H_y, mask=lookahead_mask) + H_y
        Out_ = self.norm_1(Out_)
        Out_ = self.mheda_layer.forward(H_q=Out_, H_k=H_x, H_v=H_x, mask=padding_mask) + Out_
        Out_ = self.norm_2(Out_)
        Out_ = self.ffn(Out_)
        Out = self.norm_3(Out_)  # H_y updated
        return H_x, Out, padding_mask, lookahead_mask


class Decoder(torch.nn.Module):

    def __init__(self, hidden_size: int, max_length: int, heads: int, depth: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            *[DecoderLayer(hidden_size, max_length, heads) for _ in range(depth)]
        )

    def forward(self, H_x: torch.Tensor, Y_embed: torch.Tensor, padding_mask: torch.Tensor, lookahead_mask: torch.Tensor) -> torch.Tensor:
        """
        :param H_x: (N, L, H)
        :param Y_embed: (N, L, H)
        :param padding_mask (N, L, H)
        :return: H_y: (N, L, H)
        """
        _, H_y, _, _ = self.layers((H_x, Y_embed, padding_mask, lookahead_mask))
        return H_y
