from typing import Tuple
import torch  # noqa
from pytorch_lightning import LightningModule
from tokenizers import Tokenizer  # noqa
from torch.optim.lr_scheduler import ReduceLROnPlateau  # noqa
from torchmetrics import functional as metricsF  # noqa
from torch.nn import functional as torchF  # noqa
from cleanformer.models.decoder import Decoder
from cleanformer.models.encoder import Encoder
from cleanformer.models import functional as cleanF  # noqa


class Transformer(LightningModule):  # lgtm [py/missing-call-to-init]
    def __init__(
        self,
        hidden_size: int,
        ffn_size: int,
        vocab_size: int,
        max_length: int,
        pad_token_id: int,  # noqa
        heads: int,
        depth: int,
        dropout: float,
        lr: float,  # noqa
        **kwargs  # noqa
    ):
        super().__init__()
        self.save_hyperparameters()
        self.token_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.encoder = Encoder(hidden_size, ffn_size, max_length, heads, depth, dropout)  # the encoder stack
        self.decoder = Decoder(hidden_size, ffn_size, max_length, heads, depth, dropout)  # the decoder stack
        self.register_buffer("pos_encodings", cleanF.pos_encodings(max_length, hidden_size))  # (L, H)

    def forward(
        self,
        src: torch.Tensor,
        tgt_r: torch.Tensor
    ) -> torch.Tensor:
        """
        :param src (N, 2, L)
        :param tgt_r (N, 2,  L)
        :return logits (N, |V|, H)
        """
        # --- unpack the batch --- #
        src_ids, src_key_padding_mask = (
            src[:, 0],
            src[:, 1],
        )  # (N, 2, L) -> (N, L), (N, L)
        tgt_r_ids, tgt_r_key_padding_mask = (
            tgt_r[:, 0],
            tgt_r[:, 1],
        )  # (N, 2, L) -> (N, L), (N, L)
        # --- lookup embeddings --- #
        src_embeddings = self.token_embeddings(src_ids)  # (N, L) -> (N, L, H)
        tgt_r_embeddings = self.token_embeddings(tgt_r_ids)  # (N, L) -> (N, L, H)
        # --- encode positions --- #
        src_embeddings += self.pos_encodings  # (N, L, H) + (L, H) -> (N, L, H)
        tgt_r_embeddings += self.pos_encodings  # (N, L, H) + (L, H) -> (N, L, H)
        # --- encode & decode --- #
        memory = self.encoder.forward(src_embeddings, src_key_padding_mask)  # ... -> (N, L, H)
        hidden = self.decoder.forward(
            tgt_r_embeddings, memory, tgt_r_key_padding_mask, src_key_padding_mask
        )
        # --- classify --- #
        cls = self.token_embeddings.weight  # (|V|, H) -  reuse the embeddings as the classifier
        logits = torch.einsum("...lh,vh->...vl", hidden, cls)  # (N, |V|, L)
        return logits

    @torch.no_grad()
    def decode(self,
               src: torch.Tensor,
               tgt_r: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        An implementation of autoregressive decoding (Greedy).
        """
        tgt = tgt_r.clone()
        tgt_ids, tgt_key_padding_mask = tgt[:, 0], tgt[:, 1]
        for t in range(1, self.hparams["max_length"]):
            logits = self.forward(src, tgt)  # ... -> (N, L, H)
            probabilities = torch.softmax(logits, dim=1)  # (N, |V|, L) -> (N, |V|, L)
            predictions = torch.argmax(probabilities, dim=1)  # (N, |V|, L) -> (N, L)
            tgt_ids[:, t] = torch.where(  # noqa
                tgt_ids[:, t - 1] == self.hparams["eos_token_id"],
                self.hparams["eos_token_id"],
                predictions[:, t - 1],
            )
            tgt_key_padding_mask[:, t] = torch.where(  # noqa
                tgt_ids[:, t] == self.hparams["eos_token_id"], 0, 1
            )
        return tgt, logits  # noqa

    def on_train_start(self):
        """
        Deep transformer models are often initialised with so-called "Xavier initialisation"
        https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        """
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], *args, **kwargs) -> dict:
        src, tgt_r, tgt_ids = batch
        logits = self.forward(src, tgt_r)
        loss = torchF.cross_entropy(
            logits, tgt_ids, ignore_index=self.hparams["pad_token_id"]
        )  # (N, |V|, L), (N, L) -> (1,)
        return {"loss": loss, "logits": logits.detach()}  # (N, L) -> (N,)  -> (1,)  # (N, |V|, L)

    @torch.no_grad()
    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], *args, **kwargs
    ) -> dict:
        return self.training_step(batch, *args, **kwargs)

    @torch.no_grad()
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], *args, **kwargs) -> dict:
        return self.training_step(batch, *args, **kwargs)

    # --- for optimisation --- #
    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams["lr"],
            betas=self.hparams["betas"],
            eps=self.hparams["eps"],
            weight_decay=self.hparams["weight_decay"],
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            verbose=True,
            mode=self.hparams["mode"],
            patience=self.hparams["patience"],
            cooldown=self.hparams["cooldown"],
            threshold=self.hparams["threshold"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.hparams["monitor"]},
        }
