import torch
import torch.nn as nn
from dekorde.components.mha import MultiHeadAttentionLayer


# FIXME: dropout을 적용할 떄 그 확률이.. 
class EncoderLayer(nn.Module):
    def __init__(self, max_length: int, embed_size: int, hidden_size: int, heads: int, dropout_prob: float = 0.1):
        super().__init__()
        self.embed_size = embed_size # 논문에서 512
        self.hidden_size = hidden_size # 논문에서 2048 (ffn의 hidden size)
        self.heads = heads
        self.multi_head_self_attention_layer = MultiHeadAttentionLayer(embed_size, heads, masked=False)
        self.norm_1 = nn.LayerNorm((max_length, embed_size))
        self.ffn = nn.Sequential([
            nn.Linear(embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_size)
        ])
        self.norm_2 = nn.LayerNorm((max_length, embed_size))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X_ep: torch.Tensor) -> torch.Tensor:
        """
        :param X_ep: (B, L, E)
        :return: H_all_x: (B, L, E)
        """
        out_1 = self.multi_head_self_attention_layer(X_ep, X_ep, X_ep) # (B, L, E)
        out_1 = self.dropout(out_1)
        out_1 = self.norm_1(X_ep + out_1) # (B, L, E)
        out_2 = self.ffn(out_1) # (B, L, E)
        out_2 = self.dropout(out_2)
        out = self.norm_2(out_1 + out_2) # (B, L, E)
        
        return out


class Encoder(nn.Module):
    def __init__(self, max_length: int, vocab_size: int, embed_size: int, hidden_size: int, heads:int, depth: int, dropout_prob: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, self.embed_size)
        self.encoder_layers = nn.Sequential(
            *[EncoderLayer(max_length, embed_size, hidden_size, heads) for _ in range(depth)]
        )

    def forward(self, X_ep: torch.Tensor) -> torch.Tensor:
        """
        :param X_ep: (B, L, E)
        :return: H_all_x: (B, L, E)
        """
        X_ep = self.embed(X_ep)
        return self.encoder_layers(X_ep)
