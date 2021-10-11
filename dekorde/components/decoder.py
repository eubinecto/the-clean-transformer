from typing import Tuple

import torch
import torch.nn as nn
from dekorde.components.mha import MultiHeadAttentionLayer


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, head_size: int, mask: torch.Tensor):
        self.ffn_inner_dim = 2048
        self.mask = mask

        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        # any layers to optimise?
        # masked
        self.multi_head_self_attn = MultiHeadAttentionLayer(d_model=d_model,
                                                            head_size=head_size,
                                                            is_masked=True)
        self.norm_for_self_attn = torch.nn.LayerNorm(d_model)
        # not masked
        self.multi_head_enc_dec_attn = MultiHeadAttentionLayer(d_model=d_model,
                                                               head_size=head_size,
                                                               is_masked=False)
        self.norm_for_enc_dec_attn = torch.nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.ffn_inner_dim),
            nn.ReLU(),
            nn.Linear(self.ffn_inner_dim, d_model)
        )
        self.norm_for_ffn = torch.nn.LayerNorm(d_model)

    def forward(self,
                H: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param H: ( (N, L, E), (N, L, H) )
        :return: H_all_t: (N, L, H)
        """
        H_y, H_x_out = H
        self_mha_out = self.multi_head_self_attn.forward(H_q=H_y, H_k=H_y, H_v=H_y, M=self.mask)
        self_mha_layer_out = self.norm_for_self_attn(self_mha_out + H_y)

        dec_enc_attn_out = self.multi_head_enc_dec_attn.forward(H_q=self_mha_layer_out,
                                                                H_k=H_x_out,
                                                                H_v=H_x_out)
        dec_enc_attn_layer_out = self.norm_for_enc_dec_attn(dec_enc_attn_out + self_mha_layer_out)

        ffn_out = self.ffn(dec_enc_attn_layer_out)
        ffn_layer_out = self.norm_for_ffn(ffn_out + dec_enc_attn_layer_out)

        return ffn_layer_out, H_x_out


class Decoder(torch.nn.Module):

    def __init__(self, d_model: int, head_size: int, mask: torch.Tensor, depth: int):
        super().__init__()
        self.decoder_layers = torch.nn.Sequential(
            *[DecoderLayer(d_model, head_size, mask) for _ in range(depth)]
        )

    def forward(self,
                H_y: torch.Tensor,
                H_x_out: torch.Tensor) -> torch.Tensor:
        """
        :param H_y: (N, L, E)
        :param H_x_out: (N, L, H)
        :return: H_all_t: (N, L, H)
        """
        out, _ = self.decoder_layers((H_y, H_x_out))

        return out
