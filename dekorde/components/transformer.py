import torch
from typing import Tuple
from argparse import Namespace
from torch.nn import functional as F
from dekorde.components.encoder import Encoder
from dekorde.components.decoder import Decoder
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


class Transformer(LightningModule):

    def __init__(self, hidden_size: int, vocab_size: int,
                 max_length: int, heads: int, depth: int, dropout: float, pad_token_id: int,
                 lr: float):
        super().__init__()
        self.save_hyperparameters(Namespace(hidden_size=hidden_size,
                                            vocab_size=vocab_size,
                                            max_length=max_length,
                                            heads=heads,
                                            depth=depth,
                                            dropout=dropout,
                                            pad_token_id=pad_token_id,
                                            lr=lr))
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
        :return tgt_hidden: (N, L, H)
        """
        N, _, = src_ids.size()
        # all of them are (N, L)
        positions = self.positions.repeat(N, 1)  # (L) -> (N, L)
        # --- get the embedding vectors --- #
        src_embed = self.token_embeddings(src_ids) + self.pos_embeddings(positions)  # positional encoding
        tgt_embed = self.token_embeddings(tgt_ids) + self.pos_embeddings(positions)  # positional encoding
        # --- generate the hidden vectors --- #
        src_hidden = self.encoder(src_embed, src_mask)  # (N, L, H) -> (N, L, H)
        tgt_hidden = self.decoder(src_hidden, tgt_embed, src_mask, tgt_mask)  # ... (N, L, H)
        return tgt_hidden

    def on_train_start(self) -> None:
        # this is important!
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], *args, **kwargs) -> dict:
        """
        A function for computing the loss for this batch.
        :return: a scalar tensor
        """
        X, y = batch
        src_ids, src_mask = X[:, 0, 0], X[:, 0, 1]
        tgt_ids, tgt_mask = X[:, 1, 0], X[:, 1, 1]
        tgt_hidden = self.forward(src_ids, tgt_ids, src_mask, tgt_mask)  # ... -> (N, L, H)
        # reuse the embeddings as the classifier
        cls = self.token_embeddings.weight  # (|V|, H)
        # reduce the matrices over the dimension H.
        # cross entropy of 3D input? - https://stackoverflow.com/a/63650146
        logits = torch.einsum("nlh,vh->nvl", tgt_hidden, cls)  # (N, |V|, L)
        # (N, |V|, L), (N, L) -> (N, 1) -> (1)
        loss = F.cross_entropy(logits, y, ignore_index=self.hparams['pad_token_id']).sum()
        return {
            'loss': loss
        }

    def on_train_batch_end(self, outputs: dict, *args, **kwargs) -> None:
        self.log("Train/loss", outputs['loss'], on_step=True)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], *args, **kwargs) -> dict:
        return self.training_step(batch)

    def on_validation_batch_end(self, outputs: dict, *args, **kwargs) -> None:
        self.log("Validation/loss", outputs['loss'], on_step=True)

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams['lr'])

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, 2, 2, L)
        :return: (N, L)
        """
        src_ids, src_mask = X[:, 0, 0], X[:, 0, 1]
        tgt_ids, tgt_mask = X[:, 1, 0], X[:, 1, 1]
        for time in range(1, self.hparams['max_length']):
            tgt_hidden = self.forward(src_ids, tgt_ids, src_mask, tgt_mask)  # (N, 2, 2, L) -> (N, L, H)
            cls = self.token_embeddings.weight
            logits = torch.einsum("nlh,vh->nvl", tgt_hidden, cls)  # (N, L, H) * (|V|, H) -> (N, |V|, L)
            probs = torch.softmax(logits, dim=1)  # (N,|V|, L) -> (N, |V|, L)
            indices = torch.argmax(probs, dim=1)  # (N,|V| ,L) -> (N, L)
            preds = indices[:, time]  # (N, L) -> (N, 1)
            tgt_ids[:, time] = preds
            tgt_mask[:, time] = 1
        return tgt_ids

    # just ignore these
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
