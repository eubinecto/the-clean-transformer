import torch
import torch.nn as nn
from dekorde.components.mha import MultiHeadAttentionLayer


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, head_size: int):
        self.ffn_inner_dim = 2048

        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        # any layers to optimise?
        self.multi_head_self_attention_layer = MultiHeadAttentionLayer(d_model=d_model,
                                                                       head_size=head_size,
                                                                       is_masked=False)
        self.norm_for_mha = torch.nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.ffn_inner_dim),
            nn.ReLU(),
            nn.Linear(self.ffn_inner_dim, d_model)
        )
        self.norm_for_ffn = torch.nn.LayerNorm(d_model)

    def forward(self, H_x: torch.Tensor) -> torch.Tensor:
        """
        :param H_x: (N, L, E)
        :return: H_all_x: (N, L, H)
        """

        mha_out = self.multi_head_self_attention_layer.forward(H_q=H_x, H_k=H_x, H_v=H_x)
        mha_layer_out = self.norm_for_mha(mha_out + H_x)

        ffn_out = self.ffn(mha_layer_out)
        ffn_layer_out = self.norm_for_ffn(ffn_out + mha_layer_out)

        return ffn_layer_out


class Encoder(torch.nn.Module):
    def __init__(self, d_model: int, head_size: int, depth: int):
        super().__init__()
        self.encoder_layers = torch.nn.Sequential(
            *[EncoderLayer(d_model, head_size) for _ in range(depth)]
        )

    def forward(self, H_x: torch.Tensor) -> torch.Tensor:
        """
        :param H_x: (N, L, E)
        :return: H_all_x: (N, L, H)
        """
        return self.encoder_layers(H_x)
