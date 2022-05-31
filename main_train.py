import argparse
import os
import torch  # noqa
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset  # noqa
from cleanformer import preprocess as P  # noqa
from cleanformer.fetchers import fetch_tokenizer, fetch_config, fetch_kor2eng
from cleanformer.models.transformer import Transformer
from cleanformer.paths import ROOT_DIR

# to suppress warnings - we just allow parallelism
# https://github.com/kakaobrain/pororo/issues/69#issuecomment-927564132
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    parser.add_argument("--overfit_batches", type=int, default=0)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=5)
    args = parser.parse_args()
    config = fetch_config()["transformer"]
    config.update(vars(args))
    # --- fetch a pre-trained tokenizer from wandb -- #
    tokenizer = fetch_tokenizer(config["tokenizer"])
    # --- prepare the dataloaders --- #
    kor2eng = fetch_kor2eng()
    train = TensorDataset(
        P.src(tokenizer, config["max_length"], kor2eng[0]),
        P.tgt_r(tokenizer, config["max_length"], kor2eng[0]),
        P.tgt(tokenizer, config["max_length"], kor2eng[0]),
    )
    val = TensorDataset(
        P.src(tokenizer, config["max_length"], kor2eng[1]),
        P.tgt_r(tokenizer, config["max_length"], kor2eng[1]),
        P.tgt(tokenizer, config["max_length"], kor2eng[1]),
    )
    train_dataloader = DataLoader(
        train,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        num_workers=config["num_workers"],
    )
    val_dataloader = DataLoader(
        val,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    # --- instantiate the transformer to train --- #
    transformer = Transformer(
        config["hidden_size"],
        config["ffn_size"],
        tokenizer.get_vocab_size(),  # vocab_size
        config["max_length"],
        tokenizer.pad_token_id,  # noqa
        config["heads"],
        config["depth"],
        config["dropout"],
        config["lr"],
    )
    # --- start wandb context --- #
    with wandb.init(project="cleanformer", config=config) as run:
        # --- prepare a logger (wandb) and a trainer to use --- #
        logger = WandbLogger(log_model=False)
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        trainer = Trainer(
            fast_dev_run=config["fast_dev_run"],
            check_val_every_n_epoch=config["check_val_every_n_epoch"],
            overfit_batches=config["overfit_batches"],
            max_epochs=config["max_epochs"],
            log_every_n_steps=config["log_every_n_steps"],
            gpus=torch.cuda.device_count(),
            callbacks=[lr_monitor],
            enable_checkpointing=False,
            logger=logger,
        )
        # --- start training --- #
        trainer.fit(
            model=transformer,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        # --- upload the model to wandb only if the training is properly done --- #
        if (
            not config["fast_dev_run"]
            and trainer.current_epoch == config["max_epochs"] - 1
        ):
            ckpt_path = ROOT_DIR / "transformer.ckpt"
            trainer.save_checkpoint(str(ckpt_path))
            artifact = wandb.Artifact(name="transformer", type="model", metadata=config)
            artifact.add_file(str(ckpt_path))
            run.log_artifact(artifact, aliases=["latest", config["ver"]])
            os.remove(
                str(ckpt_path)
            )  # make sure you remove it after you are done with uploading it


if __name__ == "__main__":
    main()
