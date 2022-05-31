import torch  # noqa
from cleanformer.models.ffn import FeedForward
from cleanformer.models.mha import MultiHeadAttentionLayer


class DecoderLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_size: int,
        max_length: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        # masked, multi-head self-attention layer
        self.masked_mhsa_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads, masked=True)
        # not masked, multi-head encoder-decoder attention layer
        self.mheda_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads, masked=False)
        # position-wise feed-forward network
        self.ffn = FeedForward(hidden_size, ffn_size, dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        x_key_padding_mask: torch.LongTensor,
        memory_key_padding_mask: torch.LongTensor,
    ) -> torch.Tensor:
        """
        :param: x (N, L, H)
        :param: memory (the output of the encoder) (N, L, H)
        :param: x_key_padding_mask  (N, L)
        :param: memory_key_padding_mask (N, L)
        :return: x (contextualised)
        """
        # contextualise x with itself
        x = self.masked_mhsa_layer.forward(q=x, k=x, v=x, key_padding_mask=x_key_padding_mask) + x  # residual
        # soft-align memory with respect to x
        x = (
            self.mheda_layer.forward(q=x, k=memory, v=memory, key_padding_mask=memory_key_padding_mask) + x
        )  # residual
        # apply linear transformation to each position independently but identically
        x = self.ffn(x) + x  # residual
        return x


class Decoder(torch.nn.Module):
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
        # why use ModuleList, rather than a python list?
        # A: because moduleLists are visible to Module methods but python lists are not.
        # refer to: https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
        self.layers = torch.nn.ModuleList(
            [DecoderLayer(hidden_size, ffn_size, max_length, heads, dropout) for _ in range(depth)]
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        x_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param: x (N, L, H)
        :param: memory (N, L, H)
        :param: x_key_padding_mask  (N, L)
        :param: memory_key_padding_mask (N, L)
        :return: x (contextualised): (N, L, H)
        """
        for layer in self.layers:
            x = layer.forward(x, memory, x_key_padding_mask, memory_key_padding_mask)
        return x
