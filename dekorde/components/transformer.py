import torch
from dekorde.components.encoder import Encoder
from dekorde.components.decoder import Decoder


class Transformer(torch.nn.Module):

    def __init__(self, embed_size: int, vocab_size: int, hidden_size: int, max_length: int, heads: int, depth: int):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.heads = heads
        self.depth = depth
        # any layers to optimise?
        # TODO - determine the number of embeddings
        self.token_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.pos_embeddings = torch.nn.Embedding(num_embeddings=max_length, embedding_dim=embed_size)
        self.encoder = Encoder(embed_size, hidden_size, heads, depth)  # the encoder stack
        self.decoder = Decoder(embed_size, hidden_size, heads, depth)  # the decoder stack

    def forward(self, X: torch.Tensor, Y: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L)
        :param Y: (N, L)
        :param M: (N, L, L) - attention mask
        :return: H_all_y: (N, L, H)
        """
        # TODO - h
        X_e = self.token_embeddings(...)
        X_p = self.pos_embeddings(...)
        X_ep = X_e + X_p  # positional encoding
        H_all_x = self.encoder(X_ep)  # (N, L) -> (N, L, H)
        # TODO
        Y_e = self.token_embeddings(...)
        Y_p = self.pos_embeddings(...)
        Y_ep = Y_e + Y_p  # positional encoding
        H_all_y = self.decoder(Y_ep, H_all_x, M)  # (N, L, E), (N, L, H), (N, L, L) -> (N, L, H)
        return H_all_y

    def predict(self, H_all_y: torch.Tensor) -> torch.Tensor:
        """
        :param H_all_y: (N, L, H)
        :return: S (N, L, |V|)
        """
        # S = torch.bmm(H_all_y, self.token_embeddings.weight.T.expand(N, V, H))
        S = torch.einsum("nlh,vh->nlv", H_all_y, self.token_embeddings.weight)  # (N, L, H) * (|V|, E=H) -> (N, L, |V|)

    def training_step(self, X: torch.Tensor, Y: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L) - source
        :param Y: (N, L) - target
        :param M: (???) - attention mask
        :return: loss (1,)
        """
        H_all_y = self.forward(X, Y, M)
        # TODO - compute the loss.

        raise NotImplementedError
