from typing import Tuple, List
import torch  # noqa
import wandb
from pytorch_lightning import Callback, Trainer
from tokenizers import Tokenizer  # noqa
from torchmetrics import functional as metricsF  # noqa
from cleanformer.models.transformer import Transformer


class Logger(Callback):
    """
    For logging loss, perplexity, accuracy, BLEU along with qualitative results.
    """
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    # --- for logging on batch end --- #
    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: Trainer,
        transformer: Transformer,
        out: dict,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        _, _, tgt_ids = batch
        transformer.log("train/loss", out["loss"], on_step=True, on_epoch=True)
        transformer.log("train/perplexity", torch.exp(out["loss"]), on_step=True, on_epoch=True)
        transformer.log("train/accuracy", metricsF.accuracy(out["logits"], tgt_ids), on_step=True, on_epoch=True)

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        transformer: Transformer,
        out: dict,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        _, _, tgt_ids = batch
        # logging validation metrics for each batch is unnecessary
        transformer.log("validation/loss_epoch", out["loss"], on_epoch=True)
        transformer.log("validation/perplexity_epoch", torch.exp(out["loss"]), on_epoch=True)
        transformer.log("validation/accuracy_epoch", metricsF.accuracy(out["logits"], tgt_ids), on_epoch=True)

    @torch.no_grad()
    def on_test_batch_end(
            self,
            trainer: Trainer,
            transformer: Transformer,
            out: dict,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        _, _, tgt_ids = batch
        # logging validation metrics for each batch is unnecessary
        transformer.log("test/loss_epoch", out["loss"], on_epoch=True)
        transformer.log("test/perplexity_epoch", torch.exp(out["loss"]), on_epoch=True)
        transformer.log("test/accuracy_epoch", metricsF.accuracy(out["logits"], tgt_ids), on_epoch=True)

    # --- for logging on epoch end --- #
    @torch.no_grad()
    def on_any_epoch_end(self, key: str, outputs: List[dict]):
        """
        log BLEU scores, along with qualitative infos
        """
        src_ids = torch.concat([out['src_ids'] for out in outputs], dim=0)
        tgt_hat_ids = torch.concat([out['tgt_hat_ids'] for out in outputs], dim=0)
        tgt_ids = torch.concat([out['tgt_ids'] for out in outputs], dim=0)
        losses = torch.concat([out['losses'] for out in outputs], dim=0)
        inputs = self.tokenizer.decode_batch(src_ids.tolist())
        predictions = self.tokenizer.decode_batch(tgt_hat_ids.tolist())
        answers = self.tokenizer.decode_batch(tgt_ids.tolist())
        wandb.log({
            f"{key}/examples":
            wandb.Table(columns=["input", "prediction", "answer", "losses"],
                        data=list(zip(inputs, predictions, answers, losses))),
            f"{key}/bleu": float(metricsF.bleu_score(answers, predictions))
        })

    def on_train_epoch_end(self, trainer: Trainer, transformer: Transformer) -> None:
        self.on_any_epoch_end("train", transformer.cache["train"])  # noqa

    def on_validation_epoch_end(self, trainer: Trainer, transformer: Transformer):
        self.on_any_epoch_end("validation", transformer.cache["validation"])  # noqa

    def on_test_epoch_end(self, trainer: Trainer, transformer: Transformer):
        self.on_any_epoch_end("test", transformer.cache["test"])  # noqa
