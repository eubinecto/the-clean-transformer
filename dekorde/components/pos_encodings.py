import math
import torch
import torch.nn as nn

# Regularization technique, dropout이 적용된 형태
class PositionalEncoding(nn.Module):
    def __init__(self, max_length, embed_size, droopout_prob):
        self.max_length = max_length
        self.embed_size = embed_size
        self.dropout = nn.Dropout(dropout_prob)
        self.pos_encoding = torch.zeros((max_length, embed_size)) # (L, E)

        positions = torch.arange(0, max_length).unsqueeze(1) # (L, 1) -> 이후 각 1차원마다 pos_divisions를 곱해주기 위해
        pos_divisions = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000) / embed_size)) # (E//2)

        self.pos_encoding[:, 0::2] = torch.sin(pos_divisions * positions)
        self.pos_encoding[:, 1::2] = torch.cos(pos_divisions * positions)
    
    def forward(self, x):
        x = x + self.pos_encoding[:self.max_length, :] # (L, E)
        x = self.dropout(x)
        return x