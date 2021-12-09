import torch
import numpy as np
from argparse import Namespace
from typing import Tuple
from pytorch_lightning import LightningModule
from torch.nn import functional as F


class Transformer(LightningModule):

    def __init__(self, hidden_size: int, vocab_size: int,
                 max_length: int, pad_token_id: int, heads: int, depth: int, dropout: float, lr: float):
        super().__init__()
        self.save_hyperparameters(Namespace(hidden_size=hidden_size,
                                            vocab_size=vocab_size,
                                            max_length=max_length,
                                            pad_token_id=pad_token_id,
                                            heads=heads,
                                            depth=depth,
                                            dropout=dropout,
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
    def train_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def predict_dataloader(self):
        pass


class FeedForward(torch.nn.Module):
    """
    position-wise feedforward network.
    """
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, hidden_size),
            torch.nn.Dropout(dropout)
        )

    def forward(self, hiddens: torch.Tensor):
        """
        :param hiddens: (N, L, H)
        :return: H_all: (N, L, H)
        """
        return self.layers(hiddens)


class EncoderLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, max_length: int, heads: int, dropout: float):
        super().__init__()
        # any layers to optimise?
        self.mhsa_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads, masked=False)
        self.layer_norm_1 = torch.nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, dropout)
        self.layer_norm_2 = torch.nn.LayerNorm(hidden_size)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param inputs: Tuple[src_hidden, src_mask]
        :return: src_hidden: (N, L, H)
        """
        src_hidden, src_mask = inputs
        src_hidden_ = self.mhsa_layer(Q=src_hidden, K=src_hidden, V=src_hidden,
                                      key_mask=src_mask)
        src_hidden_ = self.layer_norm_1(src_hidden_) + src_hidden_
        src_hidden_ = self.ffn(src_hidden_)
        src_hidden = self.layer_norm_2(src_hidden_) + src_hidden_  # src_hidden is now updated
        return src_hidden, src_mask


class Encoder(torch.nn.Module):
    def __init__(self, hidden_size: int, max_length: int, heads: int, depth: int, dropout: float):
        super().__init__()
        # TODO: replace this with module list
        self.encoder_layers = torch.nn.Sequential(
            *[EncoderLayer(hidden_size, max_length, heads, dropout) for _ in range(depth)]
        )

    def forward(self, src_embed: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        :param src_embed: (N, L, H)
        :param src_mask: (N, L)
        :return: input_hidden: (N, L, H)
        """
        input_hidden, _ = self.encoder_layers((src_embed, src_mask))
        return input_hidden


class DecoderLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, max_length: int, heads: int, dropout: float):
        super().__init__()
        # masked, multi-head self-attention layer.
        self.masked_mhsa_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads, masked=True)
        self.layer_norm_1 = torch.nn.LayerNorm(hidden_size)
        # not masked, multi-head encoder-decoder attention layer.
        self.mheda_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads, masked=False)
        self.layer_norm_2 = torch.nn.LayerNorm(hidden_size)
        # position-wise feed-forward network.
        self.ffn = FeedForward(hidden_size, dropout)
        self.layer_norm_3 = torch.nn.LayerNorm(hidden_size)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor])\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param inputs: (src_hidden = (N, L, H), tgt_hidden_ = (N, L, H), padding_mask (N, L))
        :return: src_hidden (as-is), tgt_hidden (updated), padding_mask (as-is)
        """
        src_hidden, tgt_hidden, src_mask, tgt_mask = inputs
        # --- contextualise
        tgt_hidden_ = self.masked_mhsa_layer(Q=tgt_hidden, K=tgt_hidden, V=tgt_hidden,
                                             key_mask=tgt_mask)
        tgt_hidden_ = self.layer_norm_1(tgt_hidden_) + tgt_hidden_
        # query = target
        # key = source
        # value = weighted average of source
        tgt_hidden_ = self.mheda_layer(Q=tgt_hidden_, K=src_hidden, V=src_hidden,
                                       key_mask=src_mask)
        tgt_hidden_ = self.layer_norm_2(tgt_hidden_) + tgt_hidden_

        tgt_hidden_ = self.ffn(tgt_hidden_)
        tgt_hidden_ = self.layer_norm_3(tgt_hidden_) + tgt_hidden_  # tgt_hidden_ updated

        # what exactly are you updating? aren't you updating the source hidden?
        return src_hidden, tgt_hidden_, src_mask, tgt_mask


class Decoder(torch.nn.Module):

    def __init__(self, hidden_size: int, max_length: int, heads: int, depth: int, dropout: float):
        super().__init__()
        # TODO: replace this with module list (No need to make things complicated)
        self.layers = torch.nn.Sequential(
            *[DecoderLayer(hidden_size, max_length, heads, dropout) for _ in range(depth)]
        )

    def forward(self, src_hidden: torch.Tensor, tgt_embed: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        :param src_hidden: (N, L, H)
        :param tgt_embed: (N, L, H)
        :param src_mask (N, L)
        :param tgt_mask (N, L)
        :return: H_y: (N, L, H)
        """
        _, tgt_hidden, _, _ = self.layers((src_hidden, tgt_embed, src_mask, tgt_mask))
        return tgt_hidden


class MultiHeadAttentionLayer(torch.nn.Module):
    """
    this could be either masked or not.
    """

    def __init__(self, hidden_size: int, max_length: int, heads: int, masked: bool):
        """
        :param hidden_size:
        :param max_length:
        :param heads:
        :param masked:
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.heads = heads
        self.masked = masked
        # hidden size must be divisible by heads.
        assert hidden_size % heads == 0
        # any layers to optimise? - four linear layers in total.
        self.W_q = torch.nn.Linear(hidden_size, hidden_size)
        self.W_k = torch.nn.Linear(hidden_size, hidden_size)
        self.W_v = torch.nn.Linear(hidden_size, hidden_size)
        self.W_o = torch.nn.Linear(hidden_size, hidden_size)  # for aggregating the multi-head outputs.
        # --- any constant tensors must be registered to a buffer --- #
        self.register_buffer("lookahead_mask", torch.tril(torch.ones(size=(max_length, max_length)), diagonal=0).long())

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        """
        :param Q: (N, L, H)
        :param K: (N, L, H)
        :param V: (N, L, H)
        :param key_mask (N, L)
        :return: hiddens (N, L, H)
        """
        # --- get batch_size --- #
        N, _, _ = Q.size()
        # --- learn patterns from Q, K, V --- #
        Q_ = self.W_q(Q)  # (N, L, H) * (H, H) -> (N, L, H)
        K_ = self.W_k(K)  # (N, L, H) * (H, H) -> (N, L, H)
        V_ = self.W_v(V)  # (N, L, H) * (H, H) -> (N, L, H)
        # transform them into multi-heads
        # --- split Q, K, V into multi-heads --- #
        # (N, L, H) -> (N, heads, L, H // heads)
        # 각 시간대 (L) 별로, 여러개의 확률분포를 허용한다 (heads).
        # 단, 나중에 모든 해드를 융합했을 때 결국 single head의 출력과 같아지도록,
        # hidden_size = hidden_size / heads 로 설정한다.
        Q_ = Q_.view(N, self.heads, self.max_length, self.hidden_size // self.heads)
        K_ = K_.view(N, self.heads, self.max_length, self.hidden_size // self.heads)
        V_ = V_.view(N, self.heads, self.max_length, self.hidden_size // self.heads)
        # compute the scaled dot product attention
        concats = self.scaled_dot_product_attention(Q_, K_, V_, key_mask)  # ... -> (N, L, H)
        hiddens = self.W_o(concats)  # (N, L, H) * (H, H) -> (N, L, H)
        return hiddens

    def scaled_dot_product_attention(self,
                                     Q: torch.Tensor,
                                     K: torch.Tensor,
                                     V: torch.Tensor,
                                     key_mask: torch.Tensor) -> torch.Tensor:
        """
         # --- einsum symbols --- #
         a = N
         b = heads
         c = H // heads
         i, j = L
        :param Q: (N, heads, L, H // heads)
        :param K: (N, heads, L, H // heads)
        :param V: (N, heads, L, H // heads)
        :param key_mask (N, L)
        :return: concats (N, L, H)
        """
        N = Q.shape[0]
        # 행렬곱 전에 미리 scale.
        # 행렬곱 이후에 스케일하면 소 잃고 외양간 고치는 격.
        Q /= np.sqrt(self.hidden_size)
        K /= np.sqrt(self.hidden_size)
        # (N, heads, L, H // heads) * (N, heads, L, H // heads) -> (N, heads, L, L)
        # sims_{abij} = \sum_{d = 1}^{d= H // heads}{Q_{abic} * K_{abjc}}
        # that is, we reduce the matrices over the "d" dimension
        sims = torch.einsum("abic,abjc->abij", Q, K)
        # the padded tokens are masked
        # (N, L) -> (N, heads, L, L)
        sims = sims.masked_fill(self.build_mask(key_mask) == 0, float("-inf"))
        # then normalise the sims to get the attention scores
        attentions = torch.softmax(sims, dim=3)  # (N, heads, L, L), normalise over keys
        # (N, heads, L, L) * (N, heads, L,  H // heads) -> (N, heads, L, H // heads)
        # contexts_{aicd} = \sum_{j = 1}^{j = L}{attentions_{acij} * V_{ajcd}}
        # that is, we reduce the matrices over the "j" dimension - the key dimension
        contexts = torch.einsum("abij,abjc->abic", attentions, V)
        # heads, H // heads -> H로 reshape하면, 결국엔 concat한 것이랑 같은 결과.
        concats = contexts.view(N, self.max_length, self.hidden_size)  # ... -> (N, L, H)
        return concats

    def build_mask(self, key_mask: torch.Tensor):
        """
        :param key_mask: (N, L)
        :return: mask (N,heads, L, L)
        """
        N, L = key_mask.size()
        # (N, L) -> (N, 1, 1, L) -> (N, heads, L, L)
        mask = key_mask.view(N, 1, 1, L)\
                       .expand(-1, self.heads, L, -1)
        # if masked, apply (logical-and it) the lookahead mask
        if self.masked:
            lookahead_mask_ = self.lookahead_mask.view(1, 1, L, L)\
                                                 .expand(N, self.heads, -1, -1)
            mask = torch.logical_and(mask, lookahead_mask_)
        return mask
