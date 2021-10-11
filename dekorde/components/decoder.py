from typing import Tuple
import torch
from dekorde.components.mha import MultiHeadAttentionLayer
from dekorde.components.ffn import FeedForward


class DecoderLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, max_length: int, heads: int, lookahead_mask: torch.Tensor):
        super().__init__()
        # masked, multi-head self-attention layer.
        self.masked_mhsa_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads, lookahead_mask)
        self.norm_1 = torch.nn.LayerNorm(hidden_size)
        # not masked, multi-head encoder-decoder attention layer.
        self.mheda_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads)
        self.norm_2 = torch.nn.LayerNorm(hidden_size)
        # position-wise feedfowrard network.
        self.ffn = FeedForward(hidden_size)
        self.norm_3 = torch.nn.LayerNorm(hidden_size)

    def forward(self, H_pair: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param H_pair: (H_x = (N, L, H), H_y = (N, L, H))
        :return: H_x (as-is), H_y (updated)
        """
        H_x, H_y = H_pair
        Out_ = self.masked_mhsa_layer.forward(H_q=H_y, H_k=H_y, H_v=H_y) + H_y  # skip connection
        Out_ = self.norm_1(Out_)
        Out_ = self.mheda_layer.forward(H_q=Out_, H_k=H_x, H_v=H_x) + Out_  # skip connection
        Out_ = self.norm_2(Out_)
        Out_ = self.ffn(Out_)
        Out = self.norm_3(Out_)  # H_y updated
        return H_x, Out


class Decoder(torch.nn.Module):

    def __init__(self, hidden_size: int, max_length: int, heads: int, depth: int, lookahead_mask: torch.Tensor):
        super().__init__()
        self.layers = torch.nn.Sequential(
            *[DecoderLayer(hidden_size, max_length, heads, lookahead_mask) for _ in range(depth)]
        )

    def forward(self, H_x: torch.Tensor, Y_embed: torch.Tensor) -> torch.Tensor:
        """
        :param H_x: (N, L, H)
        :param Y_embed: (N, L, H)
        :return: H_y: (N, L, H)
        """
        _, H_y = self.layers((H_x, Y_embed))
        return H_y
