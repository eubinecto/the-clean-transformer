from typing import Tuple, List
import torch  # noqa
from pytorch_lightning import LightningModule
from cleanformer.models.decoder import Decoder
from cleanformer.models.encoder import Encoder
from cleanformer.models import functional as cleanF  # noqa
from torch.nn import functional as torchF  # noqa
from torchmetrics import functional as metricsF  # noqa


class Transformer(LightningModule):  # lgtm [py/missing-call-to-init]
    def __init__(self, hidden_size: int, ffn_size: int,
                 vocab_size: int, max_length: int,
                 pad_token_id: int, heads: int, depth: int, dropout: float, lr: float):  # noqa
        super().__init__()
        self.save_hyperparameters()
        # --- the layers to optimise --- #
        self.token_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.encoder = Encoder(hidden_size, ffn_size, max_length, heads, depth, dropout)  # the encoder stack
        self.decoder = Decoder(hidden_size, ffn_size, max_length, heads, depth, dropout)  # the decoder stack
        # --- constant tensors --- #
        self.register_buffer("pos_encodings", cleanF.pos_encodings(max_length, hidden_size))  # (L, H)

    def forward(self, src_ids: torch.Tensor, tgt_r_ids: torch.Tensor,
                src_key_padding_mask: torch.LongTensor, tgt_r_key_padding_mask: torch.LongTensor) -> torch.Tensor:
        """
        :param src_ids (N, L)
        :param tgt_r_ids (N, L)
        :param src_key_padding_mask: (N, L)
        :param tgt_r_key_padding_mask: (N, L)
        :return hidden (N, L, H)
        """
        # --- lookup embedding vectors --- #
        src = self.token_embeddings(src_ids)  # (N, L) -> (N, L, H)
        tgt_r = self.token_embeddings(tgt_r_ids)  # (N, L) -> (N, L, H)
        # --- encode positions (the positions are broadcast-added to N) --- #
        src += self.pos_encodings  # (N, L, H) + (L, H) -> (N, L, H)
        tgt_r += self.pos_encodings  # (N, L, H) + (L, H) -> (N, L, H)
        # --- encode & decode --- #
        memory = self.encoder.forward(src, src_key_padding_mask)  # ... -> (N, L, H)
        hidden = self.decoder.forward(tgt_r, memory, tgt_r_key_padding_mask, src_key_padding_mask)  # ... (N, L, H)
        return hidden

    def step(self, src: torch.Tensor, tgt_r: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src_ids, src_key_padding_mask = src[:, 0], src[:, 1]  # (N, 2, L) -> (N, L), (N, L)
        tgt_r_ids, tgt_r_key_padding_mask = tgt_r[:, 0], tgt_r[:, 1]  # (N, 2, L) -> (N, L), (N, L)
        hidden = self.forward(src_ids, tgt_r_ids, src_key_padding_mask, tgt_r_key_padding_mask)  # ... -> (N, L, H)
        cls = self.token_embeddings.weight  # (|V|, H) -  reuse the embeddings as the classifier
        logits = torch.einsum("...lh,vh->...vl", hidden, cls)  # (N, |V|, L)
        loss = torchF.cross_entropy(logits, tgt, ignore_index=self.hparams['pad_token_id']) \
                     .sum()  # (N, |V|, L), (N, L) -> (N, 1) -> (1)
        return loss, logits

    def infer(self, src: torch.Tensor, tgt_r: torch.Tensor) -> torch.Tensor:
        """
        An implementation of autoregressive inference
        :return: (N, L)
        """
        src_ids, src_key_padding_mask = src[:, 0], src[:, 1]  # (N, 2, L) -> (N, L), (N, L)
        tgt_r_ids, tgt_r_key_padding_mask = tgt_r[:, 0], tgt_r[:, 1]  # (N, 2, L) -> (N, L), (N, L)
        for t in range(self.hparams['max_length'] - 1):
            hidden = self.forward(src_ids, tgt_r_ids, src_key_padding_mask, tgt_r_key_padding_mask)  # ... -> (N, L, H)
            cls = self.token_embeddings.weight  # (|V|, H)
            logits = torch.einsum("...lh,vh->...lv", hidden, cls)  # (N, L, H) * (|V|, H) -> (N, L, |V|)
            probs = torch.softmax(logits, dim=-1)  # (N, L, |V|) -> (N, L, |V|)
            indices = torch.argmax(probs, dim=-1)  # (N, L, |V|) -> (N, L)
            tgt_r_ids[:, t + 1] = indices[:, t]  # replace paddings with the predictions
            tgt_r_key_padding_mask[:, t + 1] = 1  # next tokens should not be ignored, so mask it
        return tgt_r_ids

    def on_train_start(self):
        # many deep transformer models are initialised with so-called "Xavier initialisation"
        # refer to: https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], *args, **kwargs) -> dict:
        """
        A function for computing the loss for this batch.
        :return: a scalar tensor containing the loss for this batch
        """
        src, tgt_r, tgt = batch
        loss, logits = self.step(src, tgt_r, tgt)
        # why detach then?
        # A: here, we need them not for computing loss, but for computing accuracies.
        # so, it's okay to detach the tensors from computation graph, thus saving some space in GPU
        # (i.e. prevent "coda out of memory error")
        # https://discuss.pytorch.org/t/cuda-out-of-memory-during-training/85014/2
        return {
            'loss': loss,
            'logits': logits.detach(),
            'tgt': tgt.detach(),
        }

    def on_train_batch_end(self, outputs: dict, *args, **kwargs):
        self.log("Train/Loss", outputs['loss'])

    def training_epoch_end(self, outputs: List[dict]) -> None:
        logits = torch.cat([out['logits'] for out in outputs],
                              dim=0)  # noqa, num_batches * (N, C) -> (num_batches * N, C)
        tgt = torch.cat([out['tgt'] for out in outputs], dim=0)  # # num_batches * (N,) -> (num_batches * N,)
        self.log("Train/Accuracy", metricsF.accuracy(logits, tgt))

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], *args, **kwargs) -> dict:
        return self.training_step(batch, *args, **kwargs)

    def on_validation_batch_end(self, outputs: dict, *args, **kwargs):
        self.log("Validation/Loss", outputs['loss'])

    def validation_epoch_end(self, outputs: List[dict]) -> None:
        logits = torch.cat([out['logits'] for out in outputs],
                              dim=0)  # noqa, num_batches * (N, C) -> (num_batches * N, C)
        tgt = torch.cat([out['tgt'] for out in outputs], dim=0)  # # num_batches * (N,) -> (num_batches * N,)
        self.log("Validation/Accuracy", metricsF.accuracy(logits, tgt))

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.hparams['lr'],
                                     betas=(0.9, 0.98), eps=1e-9)
        return {
            'optimizer': optimizer
        }
