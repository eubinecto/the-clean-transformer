import torch
import numpy as np
from abc import ABC
from argparse import Namespace
from typing import Tuple, List
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from torch.nn import functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm
from dekorde import tensors


class Transformer(LightningModule, ABC):

    def __init__(self, hidden_size: int, ffn_size: int,
                 vocab_size: int, max_length: int,
                 pad_token_id: int, heads: int, depth: int,
                 dropout: float, lr: float):
        super().__init__()
        self.save_hyperparameters(Namespace(hidden_size=hidden_size,
                                            ffn_size=ffn_size,
                                            vocab_size=vocab_size,
                                            max_length=max_length,
                                            pad_token_id=pad_token_id,
                                            heads=heads,
                                            depth=depth,
                                            dropout=dropout,
                                            lr=lr))
        # --- layers to optimise --- #
        self.token_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.encoder = Encoder(hidden_size, ffn_size, max_length, heads, depth, dropout)  # the encoder stack
        self.decoder = Decoder(hidden_size, ffn_size, max_length, heads, depth, dropout)  # the decoder stack
        # --- metrics --- #
        # we are supposed to use bleu, but let's use accuracy as the metrics to keep things simple
        self.acc_train = Accuracy(ignore_index=pad_token_id)
        self.acc_val = Accuracy(ignore_index=pad_token_id)
        self.acc_test = Accuracy(ignore_index=pad_token_id)
        # --- we register any constant tensors to the buffer instead of using to(device) --- #
        self.register_buffer("pos_encodings", tensors.pos_encodings(max_length, hidden_size))  # (L, H)

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        :param: src_ids (N, L)
        :param: tgt_ids (N, L)
        :param: src_mask (N, L)
        :param: tgt_mask (N, L)
        :return hidden: (N, L, H)
        """
        N, _ = src_ids.size()  # (N, L)
        pos_encodings = self.pos_encodings.repeat(N, 1, 1)  # (L, H) -> (N, L, H)
        src_embeddings = self.token_embeddings(src_ids) + pos_encodings  # (N, L, H) + (N, L, H) -> (N, L, H)
        tgt_embeddings = self.token_embeddings(tgt_ids) + pos_encodings  # (N, L, H) + (N, L, H) -> (N, L, H)
        memory = self.encoder(src_embeddings, src_mask)  # ... -> (N, L, H)
        hidden = self.decoder(memory, tgt_embeddings, src_mask, tgt_mask)  # ... (N, L, H)
        return hidden

    def step(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src_ids, src_mask = X[:, 0, 0], X[:, 0, 1]
        tgt_ids, tgt_mask = X[:, 1, 0], X[:, 1, 1]
        hidden = self.forward(src_ids, tgt_ids, src_mask, tgt_mask)  # ... -> (N, L, H)
        # use the embedding table as the classifier
        cls = self.token_embeddings.weight  # (|V|, H)
        # cross entropy of 3D input? - https://stackoverflow.com/a/63650146
        logits = torch.einsum("nlh,vh->nvl", hidden, cls)  # (N, |V|, L)
        # (N, |V|, L), (N, L) -> (N, 1) -> (1)
        # the lengths are different  -> pad should not be ignored
        loss = F.cross_entropy(logits, y, ignore_index=self.hparams['pad_token_id']).sum()
        return loss, logits

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, 2, 2, L)
        :return: (N, L)
        """
        src_ids, src_mask = X[:, 0, 0], X[:, 0, 1]
        tgt_ids, tgt_mask = X[:, 1, 0], X[:, 1, 1]
        for time in range(1, self.hparams['max_length']):
            hidden = self.forward(src_ids, tgt_ids, src_mask, tgt_mask)  # ... -> (N, L, H)
            cls = self.token_embeddings.weight
            logits = torch.einsum("nlh,vh->nvl", hidden, cls)  # (N, L, H) * (|V|, H) -> (N, |V|, L)
            probs = torch.softmax(logits, dim=1)  # (N,|V|, L) -> (N, |V|, L)
            indices = torch.argmax(probs, dim=1)  # (N,|V| ,L) -> (N, L)
            preds = indices[:, time]  # (N, L) -> (N, 1)
            tgt_ids[:, time] = preds
            tgt_mask[:, time] = 1
        return tgt_ids

    def on_train_start(self) -> None:
        # this is important!
        for param in tqdm(self.parameters(), desc="initialising weights..."):
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], *args, **kwargs) -> dict:
        """
        A function for computing the loss for this batch.
        :return: a scalar tensor
        """
        X, y = batch
        loss, logits = self.step(X, y)
        self.acc_train.update(logits.detach(), target=y.detach())
        return {
            'loss': loss
        }

    def on_train_batch_end(self, outputs: dict,  *args, **kwargs):
        self.log("Train/Loss", outputs['loss'])

    def training_epoch_end(self, outputs: List[dict]) -> None:
        # to see an average performance over the batches in this specific epoch
        # why detach? ->  https://discuss.pytorch.org/t/cuda-out-of-memory-during-training/85014/2
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
        # to see an average performance over the batches in this specific epoch
        # why detach? ->  https://discuss.pytorch.org/t/cuda-out-of-memory-during-training/85014/2
        avg_loss = torch.stack([output['loss'] for output in outputs]).mean()
        self.log("Validation/Average Loss", avg_loss)
        self.log("Validation/Accuracy", self.acc_val.compute())
        self.acc_val.reset()

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.hparams['lr'],
                                     betas=(0.9, 0.98), eps=1e-9)
        # what schedulers to use? : https://gaussian37.github.io/dl-pytorch-lr_scheduler/
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': "Train/Average Loss"  # better monitor the accuracy
        }


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
            torch.nn.Dropout(dropout)
        )

    def forward(self, hiddens: torch.Tensor):
        """
        :param hiddens: (N, L, H)
        :return: H_all: (N, L, H)
        """
        return self.layers(hiddens)


class EncoderLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int, max_length: int, heads: int, dropout: float):
        super().__init__()
        # any layers to optimise?
        self.mhsa_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads, masked=False)
        self.layer_norm_1 = torch.nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, ffn_size, dropout)
        self.layer_norm_2 = torch.nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param x (N, L, H)
        :param mask  (N, L)
        :return: src_hidden: (N, L, H)
        """
        # multi-head self-attention layer (mhsa)
        x = self.mhsa_layer(q=x, k=x, v=x, key_mask=mask) + x
        x = self.layer_norm_1(x)
        x = self.ffn(x) + x
        x = self.layer_norm_2(x)  # src_hidden is now updated
        return x


class Encoder(torch.nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int, max_length: int, heads: int, depth: int, dropout: float):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            EncoderLayer(hidden_size, ffn_size, max_length, heads, dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: (N, L, H)
        :param mask: (N, L)
        :return: src_hidden: (N, L, H)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int, max_length: int, heads: int, dropout: float):
        super().__init__()
        # masked, multi-head self-attention layer
        self.masked_mhsa_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads, masked=True)
        self.layer_norm_1 = torch.nn.LayerNorm(hidden_size)
        # not masked, multi-head encoder-decoder attention layer
        self.mheda_layer = MultiHeadAttentionLayer(hidden_size, max_length, heads, masked=False)
        self.layer_norm_2 = torch.nn.LayerNorm(hidden_size)
        # position-wise feed-forward network
        self.ffn = FeedForward(hidden_size, ffn_size, dropout)
        self.layer_norm_3 = torch.nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                mask: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        """
        :param: x (N, L, H)
        :param: memory (the output of the encoder) (N, L, H)
        :param: mask  (N, L)
        :param: memory_mask (N, L)
        :return: x (updated)
        """
        # --- contextualise
        x = self.masked_mhsa_layer(q=x, k=x, v=x, key_mask=mask) + x
        x = self.layer_norm_1(x)
        x = self.mheda_layer(q=x, k=memory, v=memory, key_mask=memory_mask) + x
        x = self.layer_norm_2(x)
        x = self.ffn(x) + x
        x = self.layer_norm_3(x)  # x updated
        # what exactly are you updating? aren't you updating the source hidden?
        return x


class Decoder(torch.nn.Module):

    def __init__(self, hidden_size: int, ffn_size: int, max_length: int, heads: int, depth: int, dropout: float):
        super().__init__()
        # why use ModuleList, rather than a python list?
        # A: because moduleLists are visible to Module methods but python lists are not.  (e.g. self.parameters())
        # refer to: https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
        self.layers = torch.nn.ModuleList([
            DecoderLayer(hidden_size, ffn_size, max_length, heads, dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                mask: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        """
        :param: x (N, L, H)
        :param: memory (N, L, H)
        :param: mask  (N, L)
        :param: memory_mask (N, L)
        :return: H_y: (N, L, H)
        """
        for layer in self.layers:
            x = layer(x, memory, mask, memory_mask)
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
        a query-key similarity matrix, or False if you want to apply padding mask only to the matrix
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.masked = masked
        # hidden size must be divisible by heads.
        assert hidden_size % heads == 0
        self.head_size = hidden_size // heads
        # any layers to optimise? - four linear layers in total.
        self.linear_q = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_k = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_v = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_o = torch.nn.Linear(hidden_size, hidden_size)  # for aggregating the multi-head outputs.
        # --- any constant tensors must be registered to a buffer --- #
        self.register_buffer("subsequent_mask", tensors.subsequent_mask(max_length))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        """
        :param q: (N, L, H)
        :param k: (N, L, H)
        :param v: (N, L, H)
        :param key_mask (N, L)
        :return: hiddens (N, L, H)
        """
        N, _, _ = q.size()
        # --- learn patterns from Q, K, V --- #
        q = self.linear_q(q)  # (N, L, H) * (H, H) -> (N, L, H)
        k = self.linear_k(k)  # (N, L, H) * (H, H) -> (N, L, H)
        v = self.linear_v(v)  # (N, L, H) * (H, H) -> (N, L, H)
        # transform them into multi-heads
        # --- split Q, K, V into multi-heads --- #
        # (N, L, H) -> (N, heads, L, H // heads)
        # 각 시간대 (L) 별로, 여러개의 확률분포를 허용한다 (heads).
        # 단, 나중에 모든 해드를 융합했을 때 결국 single head의 출력과 같아지도록,
        # hidden_size = hidden_size / heads 로 설정한다.
        #  (N, L, H) -> (N, L, heads, H // heads) ->  (N, heads, L, H // heads)
        q = q.view(N, -1, self.heads, self.head_size).transpose(1, 2)
        k = k.view(N, -1, self.heads, self.head_size).transpose(1, 2)
        v = v.view(N, -1, self.heads, self.head_size).transpose(1, 2)
        # compute the scaled dot product attention,
        # which outputs "contexts" (i.e. query contextualised by key and value)
        contexts = self.scaled_dot_product_attention(q, k, v, key_mask)  # ... -> (N, L, H)
        hiddens = self.linear_o(contexts)  # (N, L, H) * (H, H) -> (N, L, H)
        return hiddens

    def scaled_dot_product_attention(self,
                                     q: torch.Tensor,
                                     k: torch.Tensor,
                                     v: torch.Tensor,
                                     key_mask: torch.Tensor) -> torch.Tensor:
        """
         # --- einsum symbols --- #
         n = N
         h = heads
         i, j = L
         s = head_size
        :param q: (N, heads, L, H // heads)
        :param k: (N, heads, L, H // heads)
        :param v: (N, heads, L, H // heads)
        :param key_mask (N, L)
        :return: concats (N, L, H)
        """
        N, _, _, _ = q.size()
        # 행렬곱 전에 미리 scale.
        # 행렬곱 이후에 스케일하면 소 잃고 외양간 고치는 격.
        q /= np.sqrt(self.head_size)
        k /= np.sqrt(self.head_size)
        # (N, heads, L, H // heads) * (N, heads, L, H // heads) -> (N, heads, L, L)
        # sims_{nhij} = \sum_{d = 1}^{d= H // heads}{Q_{nhis} * K_{nhjs}}
        # that is, we reduce the matrices over the "m" dimension
        sims = torch.einsum("nhis,nhjs->nhij", q, k)
        # the padded tokens are masked
        # (N, L) -> (N, heads, L, L)
        sims = sims.masked_fill(self.build_mask(key_mask) == 0, float("-inf"))
        # then normalise the sims to get the attention scores
        attentions = torch.softmax(sims, dim=3)  # (N, heads, L, L), normalise over keys
        # (N, heads, L, L) * (N, heads, L,  H // heads) -> (N, heads, L, H // heads)
        # contexts_{nhim} = \sum_{j = 1}^{j = L}{attentions_{nhij} * V_{nhjs}}
        # that is, we reduce the matrices over the "j" dimension - the key dimension
        contexts = torch.einsum("nhij,nhjs->nhis", attentions, v)
        # (N, heads, L, H // heads) -> (N, L, heads, H // heads) -> (N, L, H)
        # why transpose?
        # A: so that we properly "concatenate" heads & H // heads dimension to hidden_size
        # why should you call contiguous after transpose?
        # A: https://stackoverflow.com/a/52229694
        contexts = contexts.transpose(1, 2) \
                           .contiguous() \
                           .view(N, -1, self.hidden_size)
        return contexts

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
            # (N, L) -> (N, 1, 1, L) -> (N, heads, L, L)
            subsequent_mask = self.subsequent_mask.view(1, 1, L, L)\
                                                  .expand(N, self.heads, -1, -1)
            # (N, heads, L, L), (N, heads, L, L) -> (N, heads, L, L)
            mask = torch.logical_and(mask, subsequent_mask)
        return mask
