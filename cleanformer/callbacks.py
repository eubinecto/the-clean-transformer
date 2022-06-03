from typing import Tuple
import torch  # noqa
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger
from tokenizers import Tokenizer  # noqa
from torchmetrics import functional as F  # noqa
from cleanformer.models.transformer import Transformer


class LogMetricsCallback(Callback):
    """
    Log metrics (loss, perplexity, accuracy)
    """

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: Transformer,
        out: dict,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        _, _, tgt = batch
        pl_module.log("Train/Loss", out["loss"], on_step=True, on_epoch=True)
        pl_module.log("Train/Perplexity", torch.exp(out["loss"]), on_step=True, on_epoch=True)
        pl_module.log("Train/Accuracy", F.accuracy(out["logits"], tgt), on_step=True, on_epoch=True)

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: Transformer,
        out: dict,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        _, _, tgt = batch
        pl_module.log("Validation/Loss_epoch", out["loss"], on_epoch=True)
        pl_module.log("Validation/Perplexity_epoch", torch.exp(out["loss"]), on_epoch=True)
        pl_module.log("Validation/Accuracy_epoch", F.accuracy(out["logits"], tgt), on_epoch=True)


class LogBLEUCallback(Callback):
    """
    Log BLEU with sample predictions
    https://docs.wandb.ai/guides/integrations/lightning#log-images-text-and-more
    """

    def __init__(self, logger: WandbLogger, tokenizer: Tokenizer):
        self.logger = logger
        self.tokenizer = tokenizer
        self.cache = dict()

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: Transformer) -> None:
        self.cache.clear()

    def on_test_epoch_start(self, trainer: Trainer, pl_module: Transformer) -> None:
        self.cache.clear()

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: Transformer,
        out: dict,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        src, tgt_r, tgt = batch  # noqa
        tgt_hat = pl_module.infer(src, tgt_r)
        self.cache["tgt"] = self.cache.get("tgt", list()) + tgt.cpu().tolist()
        self.cache["tgt_hat"] = self.cache.get("tgt_hat", list()) + tgt_hat.cpu().tolist()
        self.cache["losses"] = (
            self.cache.get("losses", list()) + out["losses"].sum(dim=1).cpu().tolist()
        )  # (N, L) -> (N,)

    @torch.no_grad()
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: Transformer,
        out: dict,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        src, tgt_r, tgt = batch  # noqa
        self.cache["tgt"] = self.cache.get("tgt", list()) + tgt.cpu().tolist()
        self.cache["tgt_hat"] = self.cache.get("tgt_hat", list()) + out["tgt_hat"].cpu().tolist()

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: Transformer) -> None:
        predictions = self.tokenizer.decode_batch(self.cache["tgt_hat"])  # (N, L) -> (N,) = list
        answers = self.tokenizer.decode_batch(self.cache["tgt"])  # (N, L) -> (N,) = list
        self.logger.log_text(
            f"Validation/samples",
            columns=["prediction", "answer", "losses"],  # noqa
            data=list(zip(predictions, answers, self.cache["losses"])),
        )
        self.logger.log_metrics({"Validation/BLEU": float(F.bleu_score(answers, predictions))})

    def on_test_epoch_end(self, trainer: Trainer, pl_module: Transformer) -> None:
        predictions = self.tokenizer.decode_batch(self.cache["tgt_hat"])  # (N, L) -> (N,) = list
        answers = self.tokenizer.decode_batch(self.cache["tgt"])  # (N, L) -> (N,) = list
        self.logger.log_text(
            f"Test/samples", columns=["prediction", "answer"], data=list(zip(predictions, answers))  # noqa
        )
        self.logger.log_metrics({"Test/BLEU": float(F.bleu_score(answers, predictions))})
