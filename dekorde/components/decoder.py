import torch
import torch.nn as nn
from dekorde.components.mha import MultiHeadAttentionLayer


class DecoderLayer(nn.Module):
    def __init__(self, max_length, embed_size: int, hidden_size: int, heads: int, dropout_prob: float = 0.1):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.heads = heads

        self.masked_multi_head_self_attention_layer = MultiHeadAttentionLayer(embed_size, heads, masked=True)
        self.norm_1 = nn.LayerNorm((max_length, embed_size))
        self.multi_head_encoder_decoder_attention_layer = MultiHeadAttentionLayer(embed_size, heads)
        self.norm_2 = nn.LayerNorm((max_length, embed_size))
        self.ffn = nn.Sequential([
            nn.Linear(embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_size)
        ])
        self.norm_3 = nn.LayerNorm((max_length, embed_size))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, Y_ep: torch.Tensor, H_all_x: torch.Tensor, Mask: torch.Tensor) -> torch.Tensor:
        """
        :param Y_ep: (N, L, E)
        :param H_all_x: (N, L, E)
        :param Mask: (N, L)
        :return: H_all_t: (N, L, H)
        """
        out_1 = self.masked_multi_head_self_attention_layer(Y_ep, H_all_x, H_all_x, Mask) # (N, L, E)
        out_1 = self.dropout(out_1)
        out_1 = self.norm_1(Y_ep + out_1) # (N, L, E)
        
        out_2 = self.multi_head_encoder_decoder_attention_layer(out_1, out_1, out_1) # (N, L, E)
        out_2 = self.dropout(out_2)
        out_2 = self.norm_2(out_1 + out_2)
        
        out_3 = self.ffn(out_2)
        out_3 = self.dropout(out_3)
        out = self.norm_3(out_2 + out_3) # (N, L, E)
        
        return out


class Decoder(nn.Module):

    def __init__(self, max_length: int, vocab_size: int, embed_size: int, hidden_size: int, heads: int, depth: int, dropout_prob: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.decoder_layers = nn.Sequential(
            *[DecoderLayer(max_length, embed_size, hidden_size, heads, dropout_prob) for _ in range(depth)]
        )
        self.linear = nn.Linear(embed_size, vocab_size)
        self.softmax = nn.Softmax()

    def forward(self, Y_ep: torch.Tensor, H_all_x: torch.Tensor, Mask: torch.Tensor) -> torch.Tensor:
        """
        :param Y_ep: (N, L, E)
        :param H_all_x: (N, L, H)
        :param M: (N, L)
        :return: H_all_t: (N, L, H)
        """
        Y_ep = self.embed(Y_ep)
        H_all_y = self.decoder_layers(Y_ep, H_all_x, Mask)
        # out = self.linear(H_all_y) # (N, L, Y_vocab_size)
        # pred = self.softmax(out) # (N, L, Y_vocab_size)
        
        return H_all_y
