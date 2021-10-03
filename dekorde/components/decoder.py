import torch
from dekorde.components.mha import MultiHeadAttentionLayer


class DecoderLayer(torch.nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, heads: int):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.heads = heads
        # any layers to optimise?
        # masked
        self.masked_multi_head_self_attention_layer = MultiHeadAttentionLayer(embed_size, hidden_size, heads)
        self.norm_1 = torch.nn.LayerNorm(...)
        # not masked
        self.multi_head_encoder_decoder_attention_layer = MultiHeadAttentionLayer(embed_size, hidden_size, heads)
        self.norm_2 = torch.nn.LayerNorm(...)
        self.ffn = torch.nn.Linear(..., ...)
        self.norm_3 = torch.nn.LayerNorm(...)

    def forward(self, Y_ep: torch.Tensor, H_all_x: torch.Tensor, M) -> torch.Tensor:
        """
        :param Y_ep: (N, L, E)
        :param H_all_x: (N, L, H)
        :param M: (???)
        :return: H_all_t: (N, L, H)
        """
        # TODO - residual connection, layer norm.
        raise NotImplementedError


class Decoder(torch.nn.Module):

    def __init__(self, embed_size: int, hidden_size: int, heads: int, depth: int):
        super().__init__()
        self.decoder_layers = torch.nn.Sequential(
            *[DecoderLayer(embed_size, hidden_size, heads) for _ in range(depth)]
        )

    def forward(self, Y_ep: torch.Tensor, H_all_x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        :param Y_ep: (N, L, E)
        :param H_all_x: (N, L, H)
        :param M: (???)
        :return: H_all_t: (N, L, H)
        """
        return self.decoder_layers(Y_ep, H_all_x, M)
