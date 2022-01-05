import torch
from tqdm import tqdm
from typing import Tuple, List
from torch.nn import functional as F
from cleanformer import functional
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule


class Transformer(LightningModule):  # lgtm [py/missing-call-to-init]
    def __init__(self, hidden_size: int, ffn_size: int,
                 vocab_size: int, max_length: int,
                 pad_token_id: int, heads: int, depth: int,
                 dropout: float, lr: float):  # noqa
        super().__init__()
        self.save_hyperparameters()
        # --- the layers to optimise --- #
        self.token_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.encoder = Encoder(hidden_size, ffn_size, max_length, heads, depth, dropout)  # the encoder stack
        self.decoder = Decoder(hidden_size, ffn_size, max_length, heads, depth, dropout)  # the decoder stack
        # --- metrics --- #
        self.acc_train = Accuracy(ignore_index=pad_token_id)
        self.acc_val = Accuracy(ignore_index=pad_token_id)
        self.acc_test = Accuracy(ignore_index=pad_token_id)
        # --- constant tensors --- #
        self.register_buffer("pos_encodings", functional.pos_encodings(max_length, hidden_size))  # (L, H)

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor,
                src_key_padding_mask: torch.LongTensor, tgt_key_padding_mask: torch.LongTensor) -> torch.Tensor:
        """
        :param src_ids (N, L)
        :param tgt_ids (N, L)
        :param src_key_padding_mask: (N, L)
        :param tgt_key_padding_mask: (N, L)
        :return hidden (N, L, H)
        """
        # --- lookup embedding vectors --- #
        src = self.token_embeddings(src_ids)  # (N, L) -> (N, L, H)
        tgt = self.token_embeddings(tgt_ids)  # (N, L) -> (N, L, H)
        # --- encode positions (the positions are broadcast-added to N) --- #
        src = src + self.pos_encodings  # (N, L, H) + (L, H) -> (N, L, H)
        tgt = tgt + self.pos_encodings  # (N, L, H) + (L, H) -> (N, L, H)
        # --- encode & decode --- #
        memory = self.encoder.forward(src, src_key_padding_mask)  # ... -> (N, L, H)
        hidden = self.decoder.forward(tgt, memory, tgt_key_padding_mask, src_key_padding_mask)  # ... (N, L, H)
        return hidden

    def step(self, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param X (N, 2, 2, L)
        :param Y (N, L)
        :return loss (,)
        """
        src_ids, src_key_padding_mask = X[:, 0, 0], X[:, 0, 1]  # (N, 2, 2, L) -> (N, L), (N, L)
        tgt_ids, tgt_key_padding_mask = X[:, 1, 0], X[:, 1, 1]  # (N, 2, 2, L) -> (N, L), (N, L)
        hidden = self.forward(src_ids, tgt_ids, src_key_padding_mask, tgt_key_padding_mask)  # ... -> (N, L, H)
        cls = self.token_embeddings.weight  # (|V|, H) -  reuse the embeddings as the classifier
        logits = torch.einsum("...lh,vh->...vl", hidden, cls)  # (N, |V|, L)
        loss = F.cross_entropy(logits, Y, ignore_index=self.hparams['pad_token_id'])\
                .sum()  # (N, |V|, L), (N, L) -> (N, 1) -> (1)
        return loss, logits

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        An implementation of auto-regressive inference
        :param X: (N, 2, 2, L)
        :return: (N, L)
        """
        src_ids, src_key_padding_mask = X[:, 0, 0], X[:, 0, 1]  # (N, 2, 2, L) -> (N, L), (N, L)
        tgt_ids, tgt_key_padding_mask = X[:, 1, 0], X[:, 1, 1]  # (N, 2, 2, L) -> (N, L), (N, L)
        for t in range(self.hparams['max_length'] - 1):
            hidden = self.forward(src_ids, tgt_ids, src_key_padding_mask, tgt_key_padding_mask)  # ... -> (N, L, H)
            cls = self.token_embeddings.weight  # (|V|, H)
            logits = torch.einsum("...lh,vh->...lv", hidden, cls)  # (N, L, H) * (|V|, H) -> (N, L, |V|)
            probs = torch.softmax(logits, dim=-1)  # (N, L, |V|) -> (N, L, |V|)
            indices = torch.argmax(probs, dim=-1)  # (N, L, |V|) -> (N, L)
            tgt_ids[:, t + 1] = indices[:, t]  # replace paddings with the predictions
            tgt_key_padding_mask[:, t + 1] = 1  # next tokens should not be ignored, so mask it
        return tgt_ids

    def on_train_start(self):
        # many deep transformer models are initialised with so-called "Xavier initialisation"
        # refer to: https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        for param in tqdm(self.parameters(), desc="initialising weights..."):
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], *args, **kwargs) -> dict:
        """
        A function for computing the loss for this batch.
        :return: a scalar tensor containing the loss for this batch
        """
        X, Y = batch
        loss, logits = self.step(X, Y)
        # why detach then?
        # A: here, we need them not for computing loss, but for computing accuracies.
        # so, it's okay to detach the tensors from computation graph, thus saving some space in GPU
        # (i.e. prevent "coda out of memory error")
        # https://discuss.pytorch.org/t/cuda-out-of-memory-during-training/85014/2
        self.acc_train.update(logits.detach(), target=Y.detach())
        return {
            'loss': loss
        }

    def on_train_batch_end(self, outputs: dict,  *args, **kwargs):
        self.log("Train/Loss", outputs['loss'])

    def training_epoch_end(self, outputs: List[dict]) -> None:
        avg_loss = torch.stack([output['loss'] for output in outputs]).mean()
        self.log("Train/Average Loss", avg_loss)
        self.log("Train/Accuracy", self.acc_train.compute())
        self.acc_train.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], *args, **kwargs) -> dict:
        X, y = batch
        loss, logits = self.step(X, y)
        self.acc_val.update(logits.detach(), target=y.detach())
        return {
            'loss': loss
        }

    def on_validation_batch_end(self, outputs: dict, *args, **kwargs):
        self.log("Validation/Loss", outputs['loss'])

    def validation_epoch_end(self, outputs: List[dict]) -> None:
        avg_loss = torch.stack([output['loss'] for output in outputs]).mean()
        self.log("Validation/Average Loss", avg_loss)
        self.log("Validation/Accuracy", self.acc_val.compute())
        self.acc_val.reset()

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.hparams['lr'],
                                     betas=(0.9, 0.98), eps=1e-9)
        return {
            'optimizer': optimizer
        }

    # ---  just ignore these (boilerplate) --- #
    def train_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def predict_dataloader(self):
        pass
    # ----------------------------------------- #


class FeedForward(torch.nn.Module):
    """
    position-wise feedforward network.
    """
    def __init__(self, hidden_size: int, ffn_size: int, dropout: float):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, ffn_size),
            torch.nn.ReLU(),
            torch.nn.Linear(ffn_size, hidden_size),
            torch.nn.Dropout(dropout),
            torch.nn.LayerNorm(hidden_size)
        )

    def forward(self, x: torch.Tensor):
        """
        :param x: (N, L, H)
        :return: x (hidden): (N, L, H)
        """
        return self.layers(x)


class EncoderLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int, max_length: int, heads: int, dropout: float):
        super().__init__()
        # not masked, multi-head self-attention layer
        self.mhsa_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads, masked=False)
        # position-wise feedforward network
        self.ffn = FeedForward(hidden_size, ffn_size, dropout)

    def forward(self, x: torch.Tensor, x_key_padding_mask: torch.LongTensor) -> torch.Tensor:
        """
        :param x (N, L, H)
        :param x_key_padding_mask (N, L)
        :return: src_hidden: (N, L, H)
        """
        # contextualised x with itself
        x = self.mhsa_layer.forward(q=x, k=x, v=x, key_padding_mask=x_key_padding_mask) + x  # residual
        # apply linear transformation to each positional identically but independently
        x = self.ffn(x) + x  # residual
        return x


class Encoder(torch.nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int, max_length: int, heads: int, depth: int, dropout: float):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            EncoderLayer(hidden_size, ffn_size, max_length, heads, dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, x_key_padding_mask: torch.LongTensor) -> torch.Tensor:
        """
        :param x: (N, L, H)
        :param x_key_padding_mask: (N, L)
        :return: x (contextualised): (N, L, H)
        """
        for layer in self.layers:
            x = layer(x, x_key_padding_mask)
        return x


class DecoderLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int, max_length: int, heads: int, dropout: float):
        super().__init__()
        # masked, multi-head self-attention layer
        self.masked_mhsa_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads, masked=True)
        # not masked, multi-head encoder-decoder attention layer
        self.mheda_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads, masked=False)
        # position-wise feed-forward network
        self.ffn = FeedForward(hidden_size, ffn_size, dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                x_key_padding_mask: torch.LongTensor, memory_key_padding_mask: torch.LongTensor) \
            -> torch.Tensor:
        """
        :param: x (N, L, H)
        :param: memory (the output of the encoder) (N, L, H)
        :param: x_key_padding_mask  (N, L)
        :param: memory_key_padding_mask (N, L)
        :return: x (contextualised)
        """
        # contextualise x with itself
        x = self.masked_mhsa_layer.forward(q=x, k=x, v=x, key_padding_mask=x_key_padding_mask) + x  # residual
        # soft-align memory with respect to x
        x = self.mheda_layer.forward(q=x, k=memory, v=memory, key_padding_mask=memory_key_padding_mask) + x  # residual
        # apply linear transformation to each position independently but identically
        x = self.ffn(x) + x  # residual
        return x


class Decoder(torch.nn.Module):

    def __init__(self, hidden_size: int, ffn_size: int, max_length: int, heads: int, depth: int, dropout: float):
        super().__init__()
        # why use ModuleList, rather than a python list?
        # A: because moduleLists are visible to Module methods but python lists are not.
        # refer to: https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
        self.layers = torch.nn.ModuleList([
            DecoderLayer(hidden_size, ffn_size, max_length, heads, dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                x_key_padding_mask: torch.Tensor, memory_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        :param: x (N, L, H)
        :param: memory (N, L, H)
        :param: x_key_padding_mask  (N, L)
        :param: memory_key_padding_mask (N, L)
        :return: x (contextualised): (N, L, H)
        """
        for layer in self.layers:
            x = layer.forward(x, memory, x_key_padding_mask, memory_key_padding_mask)
        return x


class MultiHeadAttentionLayer(torch.nn.Module):
    """
    this could be either masked or not.
    """

    def __init__(self, hidden_size: int, max_length: int, heads: int, masked: bool):
        """
        :param hidden_size:
        :param max_length:
        :param heads: the number of heads
        :param masked: set this to True if you want to apply subsequent mask as well as padding mask to
        a query-key similarity matrix, False if you want to apply only the padding mask to the matrix
        """
        super().__init__()
        assert hidden_size % heads == 0, "hidden size is not divisible by heads"
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.heads = heads
        self.masked = masked
        self.head_size = hidden_size // heads
        # --- layers to optimise --- #
        self.linear_q = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_k = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_v = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_o = torch.nn.Linear(hidden_size, hidden_size)
        self.norm = torch.nn.LayerNorm(hidden_size)
        # --- constant tensors --- #
        self.register_buffer("subsequent_mask", functional.subsequent_mask(max_length))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                key_padding_mask: torch.LongTensor) -> torch.Tensor:
        """
        :param q: (N, L, H)
        :param k: (N, L, H)
        :param v: (N, L, H)
        :param key_padding_mask (N, L)
        :return: alignments (N, L, H)
        """
        N = q.shape[0]
        # linear transformation of q, k and v
        q = self.linear_q(q)  # (N, L, H) * (H, H) -> (N, L, H)
        k = self.linear_k(k)  # (N, L, H) * (H, H) -> (N, L, H)
        v = self.linear_v(v)  # (N, L, H) * (H, H) -> (N, L, H)
        # split q, k, v into multi-heads
        q = q.view(N, self.max_length, self.heads, self.head_size)  # (N, L, H) -> (N, L, heads, head_size)
        k = k.view(N, self.max_length, self.heads, self.head_size)  # (N, L, H) -> (N, L, heads, head_size)
        v = v.view(N, self.max_length, self.heads, self.head_size)  # (N, L, H) -> (N, L, heads, head_size)
        # make q, k and v matmul-compatible
        q = q.transpose(1, 2)  # (N, L, heads, head_size) -> (N, heads, L, head_size)
        k = k.transpose(1, 2)  # (N, L, heads, head_size) -> (N, heads, L, head_size)
        v = v.transpose(1, 2)  # (N, L, heads, head_size) -> (N, heads, L, head_size)
        # key mask = key padding mask: ignore [PAD] tokens
        key_mask = key_padding_mask\
            .view(N, 1, 1, self.max_length) \
            .expand(-1, self.heads, self.max_length, -1)  # (N, L) -> (N, 1, 1, L) -> (N, heads, L, L)
        # if masked, key mask = key padding mask && key subsequent mask: ignore subsequent positions as well
        if self.masked:
            key_subsequent_mask = self.subsequent_mask\
                .view(1, 1, self.max_length, self.max_length) \
                .expand(N, self.heads, -1, -1)  # (L, L) -> (1, 1, L, L) -> (N, heads, L, L)
            key_mask = torch.logical_and(key_mask, key_subsequent_mask).long()
        # soft-align values with respect to the similarities of their keys to each query
        alignments = functional.scaled_dot_product_attention(q, k, v, key_mask)
        # concat(head_1, head_2, ... head_heads): concatenate multiple alignments
        # (N, heads, L, head_size) -> (N, L, heads, head_size) -> (N, L, H)
        concats = alignments.transpose(1, 2)\
                            .contiguous()\
                            .view(-1, self.max_length, self.hidden_size)
        # concat(head_1, head_2, ... head_heads) * W_o: aggregate alignments
        alignments = self.linear_o(concats)  # (N, L, H) * (H, H) -> (N, L, H)
        return self.norm(alignments)  # layer normalisation
