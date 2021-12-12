import torch
import numpy as np
from argparse import Namespace
from typing import Tuple, List
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from torch.nn import functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm
from enkorde import tensors


class Transformer(LightningModule):
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

    def __init__(self, implementation: str, hidden_size: int, ffn_size: int,
                 vocab_size: int, max_length: int,
                 pad_token_id: int, heads: int, depth: int,
                 dropout: float, lr: float):
        super().__init__()
        self.save_hyperparameters(Namespace(implementation=implementation,
                                            hidden_size=hidden_size,
                                            ffn_size=ffn_size,
                                            vocab_size=vocab_size,
                                            max_length=max_length,
                                            pad_token_id=pad_token_id,
                                            heads=heads,
                                            depth=depth,
                                            dropout=dropout,
                                            lr=lr))
        self.token_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        # --- choose the implementation for encoder-decoder --- #
        if implementation == EncoderDecoderTorch.name:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.enc_dec = EncoderDecoderTorch(hidden_size, ffn_size, max_length, heads, depth, dropout, device)
        elif implementation == EncoderDecoderScratch.name:
            self.enc_dec = EncoderDecoderScratch(hidden_size, ffn_size, max_length, heads, depth, dropout)
        else:
            raise ValueError(f"Invalid implementation: {implementation}")
        # --- metrics --- #
        # accuracies are rather inappropriate measure of translation quality, but let's just use them as the
        # metrics to keep it simple.
        self.acc_train = Accuracy(ignore_index=pad_token_id)
        self.acc_val = Accuracy(ignore_index=pad_token_id)
        self.acc_test = Accuracy(ignore_index=pad_token_id)

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor,
                src_key_padding_mask: torch.LongTensor, tgt_key_padding_mask: torch.LongTensor) -> torch.Tensor:
        """
        :param src_ids (N, L)
        :param tgt_ids (N, L)
        :param src_key_padding_mask: (N, L)
        :param tgt_key_padding_mask: (N, L)
        :return (N, L, H)
        """
        src = self.token_embeddings(src_ids)  # (N, L) -> (N, L, H)
        tgt = self.token_embeddings(tgt_ids)  # (N, L) -> (N, L, H)
        hidden = self.enc_dec.forward(src, tgt, src_key_padding_mask, tgt_key_padding_mask)
        return hidden

    def step(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src_ids, src_key_padding_mask = X[:, 0, 0], X[:, 0, 1]
        tgt_ids, tgt_key_padding_mask = X[:, 1, 0], X[:, 1, 1]
        hidden = self.forward(src_ids, tgt_ids, src_key_padding_mask, tgt_key_padding_mask)  # ... -> (N, L, H)
        # use the embedding table as the pre-softmax linear transformation (i.e. a token classifier)
        # "In our model, we share the same weight matrix between the two embedding layers and the pre-softmax
        # linear transformation" - pg.5
        cls = self.token_embeddings.weight  # (|V|, H)
        # To compute a cross entropy of 3D input, the logits should follow the following shape:
        # (batch_size, num_classes, max_length)
        # - https://stackoverflow.com/a/63650146
        logits = torch.einsum("nlh,vh->nvl", hidden, cls)  # (N, |V|, L)
        # (N, |V|, L), (N, L) -> (N, 1) -> (1)
        # the lengths are different  -> pad should not be ignored
        loss = F.cross_entropy(logits, y, ignore_index=self.hparams['pad_token_id']).sum()
        return loss, logits

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        auto-regressive inference
        :param X: (N, 2, 2, L)
        :return: (N, L)
        """
        src_ids, src_key_padding_mask = X[:, 0, 0], X[:, 0, 1]
        tgt_ids, tgt_key_padding_mask = X[:, 1, 0], X[:, 1, 1]
        # run an autoregressive inference
        # refer to Alammar's amazing blog post: https://jalammar.github.io/illustrated-transformer/
        # in particular, this gif: https://jalammar.github.io/images/t/transformer_decoding_2.gif
        for t in range(self.hparams['max_length'] - 1):
            hidden = self.forward(src_ids, tgt_ids, src_key_padding_mask, tgt_key_padding_mask)  # ... -> (N, L, H)
            cls = self.token_embeddings.weight  # (|V|, H)
            logits = torch.einsum("nlh,vh->nlv", hidden, cls)  # (N, L, H) * (|V|, H) -> (N, L, |V|)
            probs = torch.softmax(logits, dim=2)  # (N, L, |V|) -> (N, L, |V|)
            # To keep things simple, we just use "greedy decoding" (just choose the tokens with the highest probability)
            # Ideally, you would want a decoding strategy that is more sophisticated than that (e.g. beam search)
            # for more information on decoding strategy - refer to: https://huggingface.co/blog/how-to-generate
            pred_ids = torch.argmax(probs, dim=2)  # (N, L, |V|) -> (N, L)
            # use the pred_ids for this time step as the tgt_ids for the next time step
            tgt_ids[:, t + 1] = pred_ids[:, t]
            # next tgt_ids must not be ignored, so remove the mask for the next time step
            tgt_key_padding_mask[:, t + 1] = 0
        return tgt_ids

    def on_train_start(self):
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


class EncoderDecoderTorch(torch.nn.Module):
    name: str = "torch"

    def __init__(self, hidden_size: int, ffn_size: int, max_length: int, heads: int, depth: int, dropout: float,
                 device: torch.device):
        # --- for reference, we just use a Pytorch's implementation of transformer --- #
        super().__init__()
        self.transformer = torch.nn.Transformer(d_model=hidden_size,
                                                nhead=heads,
                                                num_encoder_layers=depth,
                                                num_decoder_layers=depth,
                                                dim_feedforward=ffn_size,
                                                dropout=dropout,
                                                batch_first=True,
                                                device=device)
        # --- register constant tensors here --- #
        self.register_buffer("pos_encodings", tensors.pos_encodings(max_length, hidden_size))  # (L, H)
        self.register_buffer("no_mask", tensors.no_mask(max_length))  # (L, L)
        self.register_buffer("subsequent_mask", tensors.subsequent_mask(max_length))  # (L, L)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_key_padding_mask: torch.Tensor, tgt_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        :param: src (N, L, H)
        :param: tgt (N, L, H)
        :param: src_key_padding_mask (N, L)
        :param: tgt_key_padding_mask (N, L)
        :return hidden: (N, L, H)
        """
        N, L, _ = src.size()
        pos_encodings = self.pos_encodings.unsqueeze(0).expand(N, L, -1)  # (L, H) -> (1, L, H) -> (N, L, H)
        # --- encode positions ---  #
        src = src + pos_encodings  # (N, L, H) + (N, L, H) -> (N, L, H)
        tgt = tgt + pos_encodings  # (N, L, H) + (N, L, H) -> (N, L, H)
        # --- attention masks --- #
        src_mask = self.no_mask.bool()
        tgt_mask = self.subsequent_mask.bool()
        # --- encode & decode --- #
        hidden = self.transformer.forward(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                                          src_key_padding_mask=src_key_padding_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask)
        return hidden


class EncoderDecoderScratch(torch.nn.Module):
    name: str = "scratch"

    def __init__(self, hidden_size: int, ffn_size: int, max_length: int, heads: int,
                 depth: int, dropout: float):
        # --- we build the encoder and the decoder from scratch --- #
        super().__init__()
        self.encoder = Encoder(hidden_size, ffn_size, max_length, heads, depth, dropout)  # the encoder stack
        self.decoder = Decoder(hidden_size, ffn_size, max_length, heads, depth, dropout)  # the decoder stack
        # --- we register any constant tensors to the buffer instead of using to(device) --- #
        # this is to follow the best practice of multi-gpu training with pytorch-lightning
        # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#init-tensors-using-type-as-and-register-buffer
        self.register_buffer("pos_encodings", tensors.pos_encodings(max_length, hidden_size))  # (L, H)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_key_padding_mask: torch.LongTensor, tgt_key_padding_mask: torch.LongTensor) -> torch.Tensor:
        """
        :param: src (N, L, H)
        :param: tgt (N, L, H)
        :param: src_key_padding_mask (N, L)
        :param: tgt_key_padding_mask (N, L)
        :return hidden: (N, L, H)
        """
        N, L, _ = src.size()  # (N, L)
        pos_encodings = self.pos_encodings.unsqueeze(0).expand(N, L, -1)  # (L, H) -> (1, L, H) -> (N, L, H)
        # --- encode positions --- #
        src = src + pos_encodings  # (N, L, H) + (N, L, H) -> (N, L, H)
        tgt = tgt + pos_encodings  # (N, L, H) + (N, L, H) -> (N, L, H)
        # --- encode & decode --- #
        memory = self.encoder.forward(src, src_key_padding_mask)  # ... -> (N, L, H)
        hidden = self.decoder.forward(tgt, memory, tgt_key_padding_mask, src_key_padding_mask)  # ... (N, L, H)
        return hidden


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

    def forward(self, x: torch.Tensor, x_key_padding_mask: torch.LongTensor) -> torch.Tensor:
        """
        :param x (N, L, H)
        :param x_key_padding_mask (N, L)
        :return: src_hidden: (N, L, H)
        """
        # multi-head self-attention layer (mhsa)
        x = self.mhsa_layer.forward(q=x, k=x, v=x, key_padding_mask=x_key_padding_mask) + x
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

    def forward(self, x: torch.Tensor, x_key_padding_mask: torch.LongTensor) -> torch.Tensor:
        """
        :param x: (N, L, H)
        :param x_key_padding_mask: (N, L)
        :return: src_hidden: (N, L, H)
        """
        for layer in self.layers:
            x = layer(x, x_key_padding_mask)
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
                x_key_padding_mask: torch.LongTensor, memory_key_padding_mask: torch.LongTensor) -> torch.Tensor:
        """
        :param: x (N, L, H)
        :param: memory (the output of the encoder) (N, L, H)
        :param: x_key_padding_mask  (N, L)
        :param: memory_mask (N, L)
        :return: x (updated)
        """
        # --- contextualise
        x = self.masked_mhsa_layer.forward(q=x, k=x, v=x, key_padding_mask=x_key_padding_mask) + x
        x = self.layer_norm_1(x)
        x = self.mheda_layer.forward(q=x, k=memory, v=memory, key_padding_mask=memory_key_padding_mask) + x
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
                x_key_padding_mask: torch.Tensor, memory_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        :param: x (N, L, H)
        :param: memory (N, L, H)
        :param: mask  (N, L)
        :param: memory_mask (N, L)
        :return: H_y: (N, L, H)
        """
        for layer in self.layers:
            x = layer(x, memory, x_key_padding_mask, memory_key_padding_mask)
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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                key_padding_mask: torch.LongTensor) -> torch.Tensor:
        """
        :param q: (N, L, H)
        :param k: (N, L, H)
        :param v: (N, L, H)
        :param key_padding_mask (N, L)
        :return: hiddens (N, L, H)
        """
        N, _, _ = q.size()
        # --- learn patterns from q, k, v --- #
        q = self.linear_q(q)  # (N, L, H) * (H, H) -> (N, L, H)
        k = self.linear_k(k)  # (N, L, H) * (H, H) -> (N, L, H)
        v = self.linear_v(v)  # (N, L, H) * (H, H) -> (N, L, H)
        # compute the scaled dot product attention,
        # which outputs "contexts" (i.e. query contextualised by key and value)
        contexts = self.multihead_scaled_dot_product_attention(q, k, v, key_padding_mask)  # ... -> (N, L, H)
        hiddens = self.linear_o(contexts)  # (N, L, H) * (H, H) -> (N, L, H)
        return hiddens

    def multihead_scaled_dot_product_attention(self,
                                               q: torch.Tensor,
                                               k: torch.Tensor,
                                               v: torch.Tensor,
                                               key_padding_mask: torch.LongTensor) -> torch.Tensor:
        """
         # --- einsum symbols --- #
         n = N
         h = heads
         q, k = the length of queries and keys
         s = head_size
        :param q: (N, L, H)
        :param k: (N, L, H)
        :param v: (N, L, H)
        :param key_padding_mask (N, L)
        :return: concats (N, L, H)
        """
        N, L, _ = q.size()
        # --- split Q, K, V into multi-heads --- #
        # (N, L, H) -> (N, heads, L, H // heads)
        # 각 시간대 (L) 별로, 여러개의 확률분포를 허용한다 (heads).
        # 단, 나중에 모든 해드를 융합했을 때 결국 single head의 출력과 같아지도록,
        # hidden_size = hidden_size / heads 로 설정한다.
        #  (N, L, H) -> (N, L, heads, H // heads) ->  (N, heads, L, H // heads)
        q = q.view(N, L, self.heads, self.head_size).transpose(1, 2)
        k = k.view(N, L, self.heads, self.head_size).transpose(1, 2)
        v = v.view(N, L, self.heads, self.head_size).transpose(1, 2)
        # 행렬곱 전에 미리 scale.
        # 행렬곱 이후에 스케일하면 소 잃고 외양간 고치는 격.
        q /= np.sqrt(self.head_size)
        k /= np.sqrt(self.head_size)
        # (N, heads, L, H // heads) * (N, heads, L, H // heads) -> (N, heads, L, L)
        # sims_{nhqk} = \sum_{d = 1}^{d= H // heads}{Q_{nhqs} * K_{nhks}}
        # that is, we reduce the matrices over the "m" dimension
        sims = torch.einsum("nhqs,nhks->nhqk", q, k)
        # the padded tokens are masked
        # (N, L) -> (N, heads, L, L)
        sims = sims.masked_fill(self.build_mask(key_padding_mask) == 1, float("-inf"))
        # then normalise the sims to get the attention scores
        attentions = torch.softmax(sims, dim=3)  # (N, heads, L, L), normalise over keys
        # (N, heads, L, L) * (N, heads, L,  H // heads) -> (N, heads, L, H // heads)
        # contexts_{nhqs} = \sum_{j = 1}^{j = L}{attentions_{nhqk} * V_{nhks}}
        # that is, we reduce the matrices over the "k" dimension - the key dimension
        contexts = torch.einsum("nhqk,nhks->nhqs", attentions, v)
        # (N, heads, L, H // heads) -> (N, L, heads, H // heads) -> (N, L, H)
        # why transpose?
        # A: so that we properly "concatenate" heads & H // heads dimension to hidden_size
        # why should you call contiguous after transpose and before view?
        # A: https://stackoverflow.com/a/52229694
        contexts = contexts.transpose(1, 2) \
                           .contiguous() \
                           .view(N, L, self.hidden_size)
        return contexts

    def build_mask(self, key_padding_mask: torch.LongTensor) -> torch.LongTensor:
        """
        combine the subsequent mask & key padding mask to build attention mask
        :param key_padding_mask: (N, L)
        :return: mask (N,heads, L, L)
        """
        N, L = key_padding_mask.size()
        # (N, L) -> (N, 1, 1, L) -> (N, heads, L, L)
        mask = key_padding_mask.view(N, 1, 1, L)\
                               .expand(-1, self.heads, L, -1)
        # if masked, apply (logical-and it) the lookahead mask
        if self.masked:
            # (L, L) -> (1, 1, L, L) -> (N, heads, L, L)
            subsequent_mask = self.subsequent_mask.view(1, 1, L, L)\
                                                  .expand(N, self.heads, -1, -1)
            # (N, heads, L, L), (N, heads, L, L) -> (N, heads, L, L)
            # why logical_or, instead of logical_and?
            # if a position is masked by any of the two masks, it must be ignored
            # even if it is not masked by the other mask
            # e.g. [0, 0, 1, 1, 1] or [0, 0, 0, 0, 1] = [0, 0, 1, 1, 1]
            mask = torch.logical_or(mask, subsequent_mask).long()
        return mask
