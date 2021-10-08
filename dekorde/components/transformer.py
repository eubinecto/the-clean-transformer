import torch
import torch.nn as nn
from dekorde.components.encoder import Encoder
from dekorde.components.decoder import Decoder
from dekorde.components.pos_encodings import PositionalEncoding


# FIXME: mask 들어갈 곳
class Transformer(nn.Module):

    def __init__(self, embed_size: int, vocab_size: int, hidden_size: int, max_length: int, heads: int, depth: int, dropout_prob: float = 0.1):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.heads = heads
        self.depth = depth

        # any layers to optimise?
        self.token_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)

        self.pos_encodings = PositionalEncoding(max_length, embed_size, dropout_prob)

        self.encoder = Encoder(max_length, vocab_size, embed_size, hidden_size, heads, depth, dropout_prob)  # the encoder stack
        self.decoder = Decoder(max_length, vocab_size, embed_size, hidden_size, heads, depth, dropout_prob)  # the decoder stack
        self.softmax = nn.SoftMax()

    def forward(self, X: torch.Tensor, Y: torch.Tensor, Mask: torch.Tensor) -> torch.Tensor:
        """
        :param X: (B, L)
        :param Y: (B, L)
        :param Mask: (B, L) - attention mask
        :return: H_all_y: (B, L, H)
        """
        X_e = self.token_embeddings(X) # (B, L, E)
        X_ep = self.pos_encodings(X_e) # (B, L, E)
        H_all_x = self.encoder(X_ep)  # (B, L, E) -> (B, L, E)

        Y_e = self.token_embeddings(Y) # (B, L, E)
        Y_ep = self.pos_encodings(Y_e) # (B, L, E)
        H_all_y = self.decoder(Y_ep, H_all_x, Mask)  # (B, L, E), (B, L, E), (B, L) -> (B, L, H)
        pred = self.predict(H_all_y) # (B, V) 여야 하는데;.. (B, L, V) 가 나와버리는 건가?
        return pred

    def predict(self, H_all_y: torch.Tensor) -> torch.Tensor:
        """
        :param H_all_y: (B, L, H)
        :return: S (B, L, |V|)
        """
        S = torch.einsum("nlh,vh->nlv", H_all_y, self.token_embeddings.weight)  # (N, L, H) * (|V|, E=H) -> (N, L, |V|)
        S = self.softmax(S)
        return S

    def training_step(self, X: torch.Tensor, Y: torch.Tensor, Mask: torch.Tensor, Label: torch.Tensor) -> torch.Tensor:
        """
        :param X: (B, L) - source
        :param Y: (B, L) - target
        :param Mask: (B, L) - attention mask
        :return: loss (B, 1)
        """
        pred = self.forward(X, Y, Mask)  # (B, V)



        return 


