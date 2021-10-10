import torch
from dekorde.components.encoder import Encoder
from dekorde.components.decoder import Decoder
from dekorde.components.pos_encoder import PositionalEncoding


class Transformer(torch.nn.Module):
    def __init__(self,
                 embed_size: int,
                 vocab_size: int,
                 hidden_size: int,
                 max_length: int,
                 heads: int,
                 depth: int):

        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.heads = heads
        self.depth = depth
        # any layers to optimise?
        # TODO - determine the number of embeddings
        self.token_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.pos_encodings = PositionalEncoding(d_model=embed_size, max_len=max_length)
        self.encoder = Encoder(embed_size, hidden_size, heads, depth)  # the encoder stack
        self.decoder = Decoder(embed_size, hidden_size, heads, depth)  # the decoder stack

    def forward(self, X: torch.Tensor, Y: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L)
        :param Y: (N, L)
        :param M: (N, L, L) - attention mask
        :return: H_all_y: (N, L, H)
        """

        # ============ ENCODER ============ #
        # input embedding 구하기
        X_e = self.token_embeddings(X)

        # positional encoding 구하기
        # + encoder input 구하기
        # 이걸하려면 임베딩된 벡터가 있어야 할 것 같아서 X_e를 들고온다.
        encoder_input = self.pos_encodings(X_e)

        # encoder stack output (hidden vector)
        H_all_x = self.encoder(encoder_input)  # (N, L) -> (N, L, H)

        # ============ DECODER ============ #
        # output embedding 구하기 (shifted right)
        Y_e = self.token_embeddings(Y)

        # positional encoding 구하기
        # + decoder input
        decoder_input = self.pos_encodings(Y_e)

        # 잠깐만... M이 필요한가?
        H_all_y = self.decoder(decoder_input, H_all_x, M)  # (N, L, E), (N, L, H), (N, L, L) -> (N, L, H)

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
