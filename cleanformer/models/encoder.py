import torch  # noqa
from cleanformer.models.ffn import FeedForward
from cleanformer.models.mha import MultiHeadAttentionLayer


class EncoderLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_size: int,
        max_length: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        # not masked, multi-head self-attention layer
        self.mhsa_layer = MultiHeadAttentionLayer(
            hidden_size, max_length, heads, masked=False
        )
        # position-wise feedforward network
        self.ffn = FeedForward(hidden_size, ffn_size, dropout)

    def forward(
        self, x: torch.Tensor, x_key_padding_mask: torch.LongTensor
    ) -> torch.Tensor:
        """
        :param x (N, L, H)
        :param x_key_padding_mask (N, L)
        :return: src_hidden: (N, L, H)
        """
        # contextualised x with itself
        x = (
            self.mhsa_layer.forward(q=x, k=x, v=x, key_padding_mask=x_key_padding_mask)
            + x
        )  # residual
        # apply linear transformation to each positional identically but independently
        x = self.ffn(x) + x  # residual
        return x


class Encoder(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_size: int,
        max_length: int,
        heads: int,
        depth: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                EncoderLayer(hidden_size, ffn_size, max_length, heads, dropout)
                for _ in range(depth)
            ]
        )

    def forward(
        self, x: torch.Tensor, x_key_padding_mask: torch.LongTensor
    ) -> torch.Tensor:
        """
        :param x: (N, L, H)
        :param x_key_padding_mask: (N, L)
        :return: x (contextualised): (N, L, H)
        """
        for layer in self.layers:
            x = layer(x, x_key_padding_mask)
        return x
