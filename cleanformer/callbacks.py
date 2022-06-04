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
        # logging validation metrics for each batch is unnecessary
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

    def on_train_epoch_start(self, *args, **kwargs):
        self.cache.pop("Train", None)
        self.cache["Train"] = dict()

    def on_validation_epoch_start(self, *args, **kwargs):
        self.cache.pop("Validation", None)
        self.cache["Validation"] = dict()

    def on_test_epoch_start(self, *args, **kwargs):
        self.cache.pop("Test", None)
        self.cache["Test"] = dict()
        
    def on_any_batch_end(self, key: str, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                         losses: torch.Tensor,
                         transformer: Transformer):
        """
        cache any data needed for logging
        """
        src, tgt_r, tgt_ids = batch
        tgt_hat_ids = transformer.infer(src, tgt_r)
        self.cache[key]["src_ids"] = self.cache[key].get("src_ids", list()) + src[:, 0].cpu().tolist()
        self.cache[key]["tgt_ids"] = self.cache[key].get("tgt_ids", list()) + tgt_ids.cpu().tolist()
        self.cache[key]["tgt_hat_ids"] = self.cache[key].get("tgt_hat_ids", list()) + tgt_hat_ids.cpu().tolist()
        self.cache[key]["losses"] = self.cache[key].get("losses", list()) + losses.cpu().tolist()

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
        self.on_any_batch_end("Train", batch, out['losses'].sum(dim=1), pl_module)
        
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
        self.on_any_batch_end("Validation", batch, out['losses'].sum(dim=1), pl_module)

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
        self.on_any_batch_end("Test", batch, out['losses'].sum(dim=1), pl_module)
        
    def on_any_epoch_end(self, key: str):
        """
        log BLEU scores, along with qualitative infos
        """
        inputs = self.tokenizer.decode_batch(self.cache[key]['src_ids'])
        predictions = self.tokenizer.decode_batch(self.cache[key]['tgt_hat_ids'])
        answers = self.tokenizer.decode_batch(self.cache[key]['tgt_ids'])
        losses = self.cache[key]['losses']
        self.logger.log_text(
            f"{key}/Samples",
            columns=["input", "prediction", "answer", "losses"],
            data=list(zip(inputs, predictions, answers, losses)),
        )
        self.logger.log_metrics({f"{key}/BLEU": float(F.bleu_score(answers, predictions))})

    def on_train_epoch_end(self, *args, **kwargs):
        self.on_any_epoch_end("Train")

    def on_validation_epoch_end(self, *args, **kwargs):
        self.on_any_epoch_end("Validation")
        
    def on_test_epoch_end(self, *args, **kwargs):
        self.on_any_epoch_end("Test")
