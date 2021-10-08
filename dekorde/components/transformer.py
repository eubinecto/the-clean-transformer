import torch
from dekorde.components.encoder import Encoder
from dekorde.components.decoder import Decoder
from torch.nn import functional as F


class Transformer(torch.nn.Module):

    def __init__(self, hidden_size: int, vocab_size: int,
                 max_length: int, heads: int, depth: int, M: torch.Tensor):
        super().__init__()
        # --- hyper parameters --- #
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.heads = heads
        self.depth = depth
        # --- layers to optimise --- #
        self.token_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.pos_embeddings = torch.nn.Embedding(num_embeddings=max_length, embedding_dim=hidden_size)
        self.encoder = Encoder(hidden_size, max_length, heads, depth)  # the encoder stack
        self.decoder = Decoder(hidden_size, max_length, heads, depth, M)  # the decoder stack
        # TODO - define the shape of this layer.

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L)
        :param Y: (N, L)
        :return H_y: (N, L, H)
        """
        N = X.shape[0]
        pos_indices = torch.arange(self.max_length).expand(N, self.max_length, self.max_length)
        X_embed = self.token_embeddings(X) + self.pos_embeddings(pos_indices)  # positional encoding
        Y_embed = self.token_embeddings(Y) + self.pos_embeddings(pos_indices)  # positional encoding
        H_x = self.encoder.forward(X_embed)  # (N, L, H) -> (N, L, H)
        H_y = self.decoder.forward(H_x, Y_embed)  # (N, L, H) -> (N, L, H)
        return H_y

    def predict(self, X, Y: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L)
        :param Y: (N, L). right-shifted.
        :return: Y_hat (N, L, |V|)
        """
        H_y = self.forward(X, Y)  # (N, L, H)
        W_hy = self.token_embeddings.weight  # (|V|, H)
        Logits = torch.einsum("nlh,vh->nlv", H_y, W_hy)  # (N, L, |V|)
        Y_hat = torch.softmax(Logits, dim=1)
        return Y_hat

    def training_step(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        A function for computing the loss for this batch.
        :param X: (N, L) - source
        :param Y: (N, 2, L) - target. (0=right shifted, 1=as-is)
        :return: loss (1,)
        """
        Y_r = Y[:, 0]  # right-shifted Y
        Y = Y[:, 1]  # Y, as-is.
        Y_hat = self.predict(X, Y_r)
        loss = F.cross_entropy(Y_hat, Y).sum()
        return loss
