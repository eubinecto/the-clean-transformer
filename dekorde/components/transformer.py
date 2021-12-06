import torch
from torch.nn import functional as F
from dekorde.components.encoder import Encoder
from dekorde.components.decoder import Decoder


class Transformer(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int,
                 max_length: int, heads: int, depth: int, dropout: float):
        super().__init__()
        self.vocab_size = vocab_size
        # --- layers to optimise --- #
        self.token_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.pos_embeddings = torch.nn.Embedding(num_embeddings=max_length, embedding_dim=hidden_size)
        self.encoder = Encoder(hidden_size, max_length, heads, depth, dropout)  # the encoder stack
        self.decoder = Decoder(hidden_size, max_length, heads, depth, dropout)  # the decoder stack
        # --- we register any constant tensors to the buffer instead of using to(device) --- #
        self.register_buffer("positions", torch.arange(max_length))  # (L)

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        :param src_ids  (N, L)
        :param tgt_ids  (N, L)
        :param src_mask (N, L)
        :param tgt_mask (N, L)
        :return tgt_hidden: (N, L, H)
        """
        N, _ = src_ids.size()
        positions = self.positions.repeat(N, 1)  # (L) -> (N, L)
        # --- get the embedding vectors --- #
        src_embed = self.token_embeddings(src_ids) + self.pos_embeddings(positions)  # positional encoding
        tgt_embed = self.token_embeddings(tgt_ids) + self.pos_embeddings(positions)  # positional encoding
        # --- generate the hidden vectors --- #
        src_hidden = self.encoder(src_embed, src_mask)  # (N, L, H) -> (N, L, H)
        tgt_hidden = self.decoder(src_hidden, tgt_embed, src_mask, tgt_mask)  # ... (N, L, H)
        return tgt_hidden

    def training_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        A function for computing the loss for this batch.
        :param inputs: (N, 2, 2, L) - 0: input_ids / 1: attention_mask
        :param targets: (N, L) - target
        :return: a scalar tensor
        """
        # all of them are (N, L)
        src_ids, src_mask = inputs[:, 0], inputs[:, 1]
        tgt_ids, tgt_mask = targets[:, 0, 0], targets[:, 0, 1]
        tgt_y_ids, tgt_y_mask = targets[:, 1, 0], targets[:, 1, 1]
        tgt_hidden = self.forward(src_ids, tgt_ids, src_mask, tgt_mask)  # ... -> (N, L, H)
        cls = self.token_embeddings.weight  # (|V|, H)
        # reduce the matrices over the dimension H.
        # cross entropy of 3D input? - https://stackoverflow.com/a/63650146
        logits = torch.einsum("nlh,vh->nvl", tgt_hidden, cls)  # (N, |V|, L)
        loss = F.cross_entropy(logits, tgt_y_ids).sum()  # (N, |V|, L), (N, L) -> (N, 1) -> (1)
        return loss
