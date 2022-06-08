from typing import Tuple
import torch  # noqa
import wandb
from pytorch_lightning import Callback, Trainer
from tokenizers import Tokenizer  # noqa
from torch.utils.data import DataLoader  # noqa
from torchmetrics import functional as metricsF  # noqa
from torch.nn import functional as torchF  # noqa
from cleanformer.models.transformer import Transformer


class LogCallback(Callback):
    """
    For logging loss, perplexity, accuracy, BLEU along with qualitative results.
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.cache = {"train": dict(), "validation": dict(), "test": dict()}

    def on_train_epoch_start(self, *args, **kwargs) -> None:
        self.cache['train'].clear()
        self.cache['train']['batches'] = list()

    def on_validation_epoch_start(self, *args, **kwargs) -> None:
        self.cache['validation'].clear()
        self.cache['validation']['batches'] = list()

    def on_test_epoch_start(self, *args, **kwargs) -> None:
        self.cache['test'].clear()
        self.cache['test']['batches'] = list()

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: Trainer,
        transformer: Transformer,
        out: dict,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *args,
        **kwargs,
    ) -> None:
        _, _, tgt_ids = batch
        transformer.log("train/loss", out["loss"], on_step=True, on_epoch=True)
        transformer.log("train/perplexity", torch.exp(out["loss"]), on_step=True, on_epoch=True)
        transformer.log(
            "train/accuracy",
            metricsF.accuracy(out["logits"], tgt_ids, ignore_index=transformer.hparams["pad_token_id"]),
            on_step=True,
            on_epoch=True,
        )
        self.cache["train"]["batches"].append(batch)

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        transformer: Transformer,
        out: dict,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *args,
        **kwargs,
    ) -> None:
        # logging validation metrics for each batch is unnecessary
        _, _, tgt_ids = batch
        transformer.log("validation/loss_epoch", out["loss"], on_epoch=True)
        transformer.log("validation/perplexity_epoch", torch.exp(out["loss"]), on_epoch=True)
        transformer.log(
            "validation/accuracy_epoch",
            metricsF.accuracy(out["logits"], tgt_ids, ignore_index=transformer.hparams["pad_token_id"]),
            on_epoch=True,
        )
        self.cache["validation"]["batches"].append(batch)

    @torch.no_grad()
    def on_test_batch_end(
        self,
        trainer: Trainer,
        transformer: Transformer,
        out: dict,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *args,
        **kwargs,
    ) -> None:
        _, _, tgt_ids = batch
        transformer.log("test/loss_epoch", out["loss"], on_epoch=True)
        transformer.log("test/perplexity_epoch", torch.exp(out["loss"]), on_epoch=True)
        transformer.log(
            "test/accuracy_epoch",
            metricsF.accuracy(out["logits"], tgt_ids, ignore_index=transformer.hparams["pad_token_id"]),
            on_epoch=True,
        )
        self.cache["test"]["batches"].append(batch)

    # --- for logging on epoch end --- #
    def on_any_epoch_end(self, key: str, transformer: Transformer):
        """
        log BLEU scores, along with qualitative infos
        """
        inputs = list()
        answers = list()
        predictions = list()
        losses = list()
        for batch in self.cache[key]['batches']:
            src, tgt_r, tgt_ids = batch
            tgt_hat_ids, logits = transformer.infer(src, tgt_r)
            inputs += self.tokenizer.decode_batch(src[:, 0].cpu().tolist())
            answers += self.tokenizer.decode_batch(tgt_ids.cpu().tolist())
            predictions += self.tokenizer.decode_batch(tgt_hat_ids.cpu().tolist())
            losses += (
                torchF.cross_entropy(
                    logits, tgt_ids, ignore_index=transformer.hparams["pad_token_id"], reduction="none"
                )
                .mean(dim=-1)
                .cpu()
                .tolist()
            )  # (N, L) -> (N,) -> list
        wandb.log(
            {
                f"{key}/examples": wandb.Table(
                    columns=["input", "prediction", "answer", "losses"],
                    data=list(zip(inputs, predictions, answers, losses)),
                ),
                f"{key}/bleu_epoch": metricsF.bleu_score(
                    answers,
                    predictions,
                    n_gram=transformer.hparams["n_gram"],
                    smooth=transformer.hparams["smooth"],
                ),
            }
        )

    def on_train_epoch_end(self, trainer: Trainer, transformer: Transformer):
        self.on_any_epoch_end("train", transformer)

    def on_validation_epoch_end(self, trainer: Trainer, transformer: Transformer):
        self.on_any_epoch_end("validation", transformer)

    def on_test_epoch_end(self, trainer: Trainer, transformer: Transformer):
        self.on_any_epoch_end("test", transformer)
