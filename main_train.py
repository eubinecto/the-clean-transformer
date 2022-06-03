import argparse
import os
import shutil
import torch  # noqa
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset  # noqa
from cleanformer import preprocess as P  # noqa
from cleanformer.callbacks import LogMetricsCallback, LogBLEUCallback
from cleanformer.fetchers import fetch_tokenizer, fetch_config, fetch_kor2eng
from cleanformer.models.transformer import Transformer
from cleanformer.paths import WANDB_DIR

# to suppress warnings - we just allow parallelism
# https://github.com/kakaobrain/pororo/issues/69#issuecomment-927564132
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def main():
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group("required arguments")
    required.add_argument("--max_epochs", type=int, required=True)
    required.add_argument("--save_top_k", type=int, required=True)
    required.add_argument("--save_on_train_epoch_end", type=int, required=True)
    required.add_argument("--every_n_epochs", type=int, required=True)
    required.add_argument("--log_every_n_steps", type=int, required=True)
    required.add_argument("--check_val_every_n_epoch", type=int, required=True)
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument("--fast_dev_run", action="store_true", default=False)
    optional.add_argument("--overfit_batches", type=int, default=0.0)
    optional.add_argument("--limit_train_batches", type=int, default=1.0)
    optional.add_argument("--limit_val_batches", type=int, default=1.0)
    optional.add_argument("--num_workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    config = fetch_config()["transformer"]
    config.update(vars(args))
    # --- fetch a pre-trained tokenizer from wandb -- #
    tokenizer = fetch_tokenizer(config["tokenizer"])
    # --- prepare the dataloaders --- #
    train, val, _ = fetch_kor2eng(tokenizer.kor2eng)  # noqa
    train = TensorDataset(
        P.src(tokenizer, config["max_length"], train),
        P.tgt_r(tokenizer, config["max_length"], train),
        P.tgt(tokenizer, config["max_length"], train),
    )
    val = TensorDataset(
        P.src(tokenizer, config["max_length"], val),
        P.tgt_r(tokenizer, config["max_length"], val),
        P.tgt(tokenizer, config["max_length"], val),
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
    with wandb.init(project="cleanformer", config=config, tags=[__file__]):
        # --- prepare a logger (wandb) and a trainer to use --- #
        logger = WandbLogger(log_model="all", save_dir=WANDB_DIR)
        trainer = Trainer(
            logger=logger,
            fast_dev_run=config["fast_dev_run"],
            limit_train_batches=config["limit_train_batches"],
            limit_val_batches=config["limit_val_batches"],
            check_val_every_n_epoch=config["check_val_every_n_epoch"],
            overfit_batches=config["overfit_batches"],
            log_every_n_steps=config["log_every_n_steps"],
            max_epochs=config["max_epochs"],
            gpus=torch.cuda.device_count(),
            callbacks=[
                ModelCheckpoint(
                    verbose=True,
                    monitor=config["monitor"],
                    mode=config["mode"],
                    save_top_k=config["save_top_k"],
                    every_n_epochs=config["every_n_epochs"],
                    save_on_train_epoch_end=config["save_on_train_epoch_end"],
                ),
                LearningRateMonitor(logging_interval="epoch"),
                LogMetricsCallback(),
                LogBLEUCallback(logger, tokenizer),
            ],
        )
        # --- start training --- #
        trainer.fit(
            model=transformer,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
    # sweep local logs after uploading is done
    shutil.rmtree(WANDB_DIR)


if __name__ == "__main__":
    main()
