import torch
from dekorde.components.encoder import Encoder
from dekorde.components.decoder import Decoder
from torch.nn import functional as F


class Transformer(torch.nn.Module):

    def __init__(self, hidden_size: int, vocab_size: int,
                 max_length: int, heads: int, depth: int, start_token_id: int, lookahead_mask: torch.Tensor,
                 device: torch.device):
        super().__init__()
        # --- hyper parameters --- #
        self.max_length = max_length
        self.start_token_id = start_token_id
        self.lookahead_mask = lookahead_mask
        self.device = device
        # --- layers to optimise --- #
        self.token_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.pos_embeddings = torch.nn.Embedding(num_embeddings=max_length, embedding_dim=hidden_size)
        self.encoder = Encoder(hidden_size, max_length, heads, depth)  # the encoder stack
        self.decoder = Decoder(hidden_size, max_length, heads, depth, lookahead_mask)  # the decoder stack
        self.to(device)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, 2, L)
        :param Y: (N, L)
        :return H_y: (N, L, H)
        """
        N = X.shape[0]
        X_ids = X[:, 0]
        padding_mask = X[:, 1]
        pos_indices = torch.arange(self.max_length).expand(N, self.max_length).to(self.device)
        # --- get the embedding vectors --- #
        pos_embed = self.pos_embeddings(pos_indices)
        X_embed = self.token_embeddings(X_ids) + pos_embed  # positional encoding
        Y_embed = self.token_embeddings(Y) + pos_embed  # positional encoding
        # --- generate the hidden vectors --- #
        H_x = self.encoder(X_embed)  # (N, L, H) -> (N, L, H)
        H_y = self.decoder(H_x, Y_embed)  # (N, L, H), (N, L, H) -> (N, L, H)
        return H_y

    def training_step(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        A function for computing the loss for this batch.
        :param X: (N, L) - source
        :param Y: (N, 2, L) - target
        :return: loss (1,)
        """
        Y_l = Y[:, 0]  # starts from " [SOS]", ends just before the last character.
        Y_r = Y[:, 1]  # starts after "[SOS]", ends with the last character.
        H_y = self.forward(X, Y_l)  # ... -> (N, L, H)
        W_hy = self.token_embeddings.weight  # (|V|, H)
        # reduce the matrices over the dimension H.
        # cross entropy of 3D input? - https://stackoverflow.com/a/63650146
        logits = torch.einsum("abc,dc->adb", H_y, W_hy)  # (N, |V|, L)
        loss = F.cross_entropy(logits, Y_r)  # (N, |V|, L), (N, L) -> (N, 1)
        loss = loss.sum()  # (N, 1) -> (1,)
        return loss

    def infer(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L), a batch of input_ids
        :return Y: (N, L), a batch of input_ids
        """
        N, L = X.shape
        pos_indices = torch.arange(self.max_length).expand(N, self.max_length).to(self.device)
        # --- get the embedding vectors --- #
        pos_embed = self.pos_embeddings(pos_indices)
        X_embed = self.token_embeddings(X) + pos_embed
        H_x = self.encoder(X_embed)  # ... -> (N, L, H)
        # Y = torch.zeros(size=(N, L)).long().to(self.device)
        # Y = torch.ones(size=(N, L)).long().to(self.device)  # ones를 넣으면, 1을 정답으로 생각하려나?
        Y = torch.full(size=(N, L), fill_value=410).to(self.device)  # 긕
        # Y = torch.full(size=(N, L), fill_value=411)
        Y[:, 0] = self.start_token_id  # (N, L)
        W_hy = self.token_embeddings.weight  # (|V|, H)

        for time in range(1, L):
            # what do we do here?
            Y_embed = self.token_embeddings(Y) + pos_embed
            H_y = self.decoder(H_x, Y_embed)  # (N, L, H), (N, L, H) -> (N, L, H)
            logits = torch.einsum("abc,dc->abd", H_y, W_hy)  # (N, L, H) * (|V|, H) -> (N, L, |V|)
            probs = torch.softmax(logits, dim=2)
            indices = torch.argmax(probs, dim=2)
            predicted_token_ids = indices[:, time]  # (N, L) -> (N, 1)
            Y[:, time] = predicted_token_ids
        return Y
