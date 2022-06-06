from typing import Tuple, List
import torch  # noqa
import wandb
from pytorch_lightning import Callback, Trainer
from tokenizers import Tokenizer  # noqa
from torchmetrics import functional as metricsF  # noqa
from cleanformer.models.transformer import Transformer


class LogCallback(Callback):
    """
    For logging loss, perplexity, accuracy, BLEU along with qualitative results.
    """
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.cache = {"train": dict(), "validation": dict(), "test": dict()}

    def on_train_start(self, *args, **kwargs) -> None:
        self.cache["train"].clear()

    def on_validation_start(self, *args, **kwargs) -> None:
        self.cache["validation"].clear()

    def on_test_start(self, *args, **kwargs) -> None:
        self.cache["test"].clear()

    def on_any_batch_end(self, key: str, transformer: Transformer,
                         src: torch.Tensor, tgt_r: torch.Tensor, tgt_ids: torch.Tensor, losses: List[float]) -> tuple:
        inputs = self.tokenizer.decode_batch(src[:, 0].cpu().tolist())
        answers = self.tokenizer.decode_batch(tgt_ids.cpu().tolist())
        predictions = self.tokenizer.decode_batch(transformer.infer(src, tgt_r).cpu().tolist())
        self.cache[key]["inputs"] = self.cache[key].get("inputs", list()) + inputs
        self.cache[key]["answers"] = self.cache[key].get("answers", list()) + answers
        self.cache[key]["predictions"] = self.cache[key].get("predictions", list()) + predictions
        self.cache[key]["losses"] = self.cache[key].get("losses", list()) + losses
        return answers, predictions

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: Trainer,
        transformer: Transformer,
        out: dict,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *args,
        **kwargs
    ) -> None:
        src, tgt_r, tgt_ids = batch
        transformer.log("train/loss", out["loss"], on_step=True, on_epoch=True)
        transformer.log("train/perplexity", torch.exp(out["loss"]), on_step=True, on_epoch=True)
        transformer.log("train/accuracy", metricsF.accuracy(out["logits"], tgt_ids), on_step=True, on_epoch=True)
        answers, predictions = self.on_any_batch_end("train", transformer, src, tgt_r, tgt_ids,
                                                     out['losses'].cpu().tolist())
        transformer.log("train/bleu", metricsF.bleu_score(answers, predictions), on_step=True, on_epoch=True)

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        transformer: Transformer,
        out: dict,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *args,
        **kwargs
    ) -> None:
        # logging validation metrics for each batch is unnecessary
        src, tgt_r, tgt_ids = batch
        transformer.log("validation/loss_epoch", out["loss"], on_epoch=True)
        transformer.log("validation/perplexity_epoch", torch.exp(out["loss"]), on_epoch=True)
        transformer.log("validation/accuracy_epoch", metricsF.accuracy(out["logits"], tgt_ids), on_epoch=True)
        answers, predictions = self.on_any_batch_end("validation", transformer, src, tgt_r, tgt_ids,
                                                     out['losses'].cpu().tolist())
        transformer.log("validation/bleu_epoch", metricsF.bleu_score(answers, predictions), on_epoch=True)

    @torch.no_grad()
    def on_test_batch_end(
            self,
            trainer: Trainer,
            transformer: Transformer,
            out: dict,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            *args,
            **kwargs
    ) -> None:
        src, tgt_r, tgt_ids = batch
        transformer.log("test/loss_epoch", out["loss"], on_epoch=True)
        transformer.log("test/perplexity_epoch", torch.exp(out["loss"]), on_epoch=True)
        transformer.log("test/accuracy_epoch", metricsF.accuracy(out["logits"], out['tgt_ids']), on_epoch=True)
        answers, predictions = self.on_any_batch_end("test", transformer, src, tgt_r, tgt_ids,
                                                     out['losses'].cpu().tolist())
        transformer.log("test/bleu_epoch", metricsF.bleu_score(answers, predictions), on_epoch=True)

    # --- for logging on epoch end --- #
    @torch.no_grad()
    def on_any_epoch_end(self, key: str):
        """
        log BLEU scores, along with qualitative infos
        """
        inputs = self.cache[key]['inputs']
        predictions = self.cache[key]['predictions']
        answers = self.cache[key]['answers']
        losses = self.cache[key]['losses']
        wandb.log({
            f"{key}/examples":
            wandb.Table(columns=["input", "prediction", "answer", "losses"],
                        data=list(zip(inputs, predictions, answers, losses)))
        })

    def on_train_epoch_end(self, *args, **kwargs) -> None:
        self.on_any_epoch_end("train")  # noqa

    def on_validation_epoch_end(self, *args, **kwargs):
        self.on_any_epoch_end("validation")  # noqa

    def on_test_epoch_end(self, *args, **kwargs):
        self.on_any_epoch_end("test")  # noqa
