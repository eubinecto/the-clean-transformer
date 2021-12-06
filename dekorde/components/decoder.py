import torch
from typing import Tuple
from dekorde.components.mha import MultiHeadAttentionLayer
from dekorde.components.ffn import FeedForward


class DecoderLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, max_length: int, heads: int, dropout: float):
        super().__init__()
        # masked, multi-head self-attention layer.
        self.masked_mhsa_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads, masked=True)
        self.layer_norm_1 = torch.nn.LayerNorm(hidden_size)
        # not masked, multi-head encoder-decoder attention layer.
        self.mheda_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads, masked=False)
        self.layer_norm_2 = torch.nn.LayerNorm(hidden_size)
        # position-wise feed-forward network.
        self.ffn = FeedForward(hidden_size, dropout)
        self.layer_norm_3 = torch.nn.LayerNorm(hidden_size)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor])\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param inputs: (src_hidden = (N, L, H), tgt_hidden_ = (N, L, H), padding_mask (N, L))
        :return: src_hidden (as-is), tgt_hidden (updated), padding_mask (as-is)
        """
        src_hidden, tgt_hidden, src_mask, tgt_mask = inputs
        # --- contextualise
        tgt_hidden_ = self.masked_mhsa_layer(Q=tgt_hidden, K=tgt_hidden, V=tgt_hidden,
                                             key_mask=tgt_mask)
        tgt_hidden_ = self.layer_norm_1(tgt_hidden_) + tgt_hidden_
        # query = target
        # key = source
        # value = weighted average of source
        tgt_hidden_ = self.mheda_layer(Q=tgt_hidden_, K=src_hidden, V=src_hidden,
                                       key_mask=src_mask)
        tgt_hidden_ = self.layer_norm_2(tgt_hidden_) + tgt_hidden_

        tgt_hidden_ = self.ffn(tgt_hidden_)
        tgt_hidden_ = self.layer_norm_3(tgt_hidden_) + tgt_hidden_  # tgt_hidden_ updated

        # what exactly are you updating? aren't you updating the source hidden?
        return src_hidden, tgt_hidden_, src_mask, tgt_mask


class Decoder(torch.nn.Module):

    def __init__(self, hidden_size: int, max_length: int, heads: int, depth: int, dropout: float):
        super().__init__()
        self.layers = torch.nn.Sequential(
            *[DecoderLayer(hidden_size, max_length, heads, dropout) for _ in range(depth)]
        )

    def forward(self, src_hidden: torch.Tensor, tgt_embed: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        :param src_hidden: (N, L, H)
        :param tgt_embed: (N, L, H)
        :param src_mask (N, L)
        :param tgt_mask (N, L)
        :return: H_y: (N, L, H)
        """
        _, tgt_hidden, _, _ = self.layers((src_hidden, tgt_embed, src_mask, tgt_mask))
        return tgt_hidden
