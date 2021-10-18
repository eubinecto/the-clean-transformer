from functools import reduce
from typing import Tuple, List

import torch
import torch.nn.functional as F

from dekorde.components.encoder import Encoder
from dekorde.components.decoder import Decoder
from dekorde.components.pos_encoder import PositionalEncoding


class Transformer(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 vocab_size: int,
                 max_length: int,
                 head_size: int,
                 depth: int,
                 mask: torch.Tensor):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.depth = depth
        self.mask = mask
        self.max_length = max_length
        # any layers to optimise?
        # TODO - determine the number of embeddings
        self.token_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        # self.pos_encodings = PositionalEncoding(d_model=d_model, max_len=max_length)
        self.pos_embeddings = torch.nn.Embedding(num_embeddings=max_length, embedding_dim=d_model)
        self.encoder = Encoder(d_model, head_size, depth)  # the encoder stack
        self.decoder = Decoder(d_model, head_size, mask, depth)  # the decoder stack

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L)
        :param Y: (N, L)
        :return: H_all_y: (N, L, H)
        """
        N = X.shape[0]
        pos_indices = torch.arange(self.max_length).expand(N, self.max_length)

        # ============ ENCODER ============ #
        # # input embedding 구하기
        # X_e = self.token_embeddings(X)
        #
        # # positional encoding 구하기
        # # + encoder input 구하기
        # # 이걸하려면 임베딩된 벡터가 있어야 할 것 같아서 X_e를 들고온다.
        # encoder_input = self.pos_encodings(X_e)

        encoder_input = self.token_embeddings(X) + self.pos_embeddings(pos_indices)  # positional encoding

        # encoder stack output (hidden vector)
        H_all_x = self.encoder.forward(encoder_input)  # (N, L) -> (N, L, H)

        # ============ DECODER ============ #
        # output embedding 구하기 (shifted right)
        # Y_e = self.token_embeddings(Y)
        #
        # # positional encoding 구하기
        # # + decoder input
        # decoder_input = self.pos_encodings(Y_e)
        decoder_input = self.token_embeddings(Y) + self.pos_embeddings(pos_indices)  # positional encoding

        # 잠깐만... M이 필요한가 -> 필요하긴하넹.
        # (N, L, E), (N, L, H), (N, L, L) -> (N, L, H)
        H_all_y = self.decoder.forward(decoder_input, H_all_x)

        return H_all_y

    def get_logits(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param y: (N, L)
        :param X: (N, L, H)
        :return: logits (N, L, |V|)
        """
        H_y = self.forward(X, y)  # (N, L, d_model)
        W_token_embed = self.token_embeddings.weight  # (|V|, d_model)

        logits = torch.einsum("nlh,vh->nlv", H_y, W_token_embed)  # (N, L, |V|)
        return logits

    def predict(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param y: (N, L)
        :param X: (N, L, H)
        :return: logits (N, L, |V|)
        """
        logits = self.get_logits(X, y)
        probs = torch.softmax(logits, dim=0)  # (N, L, |V|)
        preds = torch.argmax(probs, dim=-1)  # (N, L)

        return logits, probs, preds

    def training_step(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param X: (N, L) - source
        :param y: (N, L) - target
        :return: loss (1,)
        """
        y_l = y[:, 0]  # starts from "s", ends just before the last character.
        y_r = y[:, 1]  # starts after "s", ends with the last character.
        logits, probs, preds = self.predict(X, y_l)
        logits = torch.einsum('nlv->nvl', logits)

        loss = F.cross_entropy(logits, y_r)
        loss = loss.sum()

        acc = (preds == y_r).float().sum() / reduce(lambda i, j: i*j, y.size())

        return loss, acc

