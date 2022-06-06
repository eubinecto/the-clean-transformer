import argparse
import os
import shutil
import torch  # noqa
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset  # noqa
from cleanformer import preprocess as P  # noqa
from cleanformer.fetchers import fetch_tokenizer, fetch_config, fetch_kor2eng
from cleanformer.logcallback import LogCallback
from cleanformer.models.transformer import Transformer
from cleanformer.paths import WANDB_DIR

# to suppress warnings - we just allow parallelism
# https://github.com/kakaobrain/pororo/issues/69#issuecomment-927564132
os.environ["TOKENIZERS_PARALLELISM"] = "true"


parser = argparse.ArgumentParser()
required = parser.add_argument_group("required arguments")
required.add_argument("--max_epochs", type=int, required=True)
required.add_argument("--batch_size", type=int, required=True)
required.add_argument("--save_on_train_epoch_end", choices=(0, 1), type=int, required=True)
required.add_argument("--every_n_epochs", type=int, required=True)
required.add_argument("--log_every_n_steps", type=int, required=True)
required.add_argument("--check_val_every_n_epoch", type=int, required=True)
optional = parser.add_argument_group("optional arguments")
optional.add_argument("--fast_dev_run", action="store_true", default=False)
optional.add_argument("--detect_anomaly", action="store_true", default=False)
optional.add_argument("--verbose", choices=(0, 1), type=int, default=1)  # could be int or str
optional.add_argument("--overfit_batches", type=int, default=0.0)
optional.add_argument("--limit_train_batches", type=int, default=1.0)
optional.add_argument("--limit_val_batches", type=int, default=1.0)
optional.add_argument("--max_depth", type=int, default=4)
optional.add_argument("--num_workers", type=int, default=os.cpu_count())
args = parser.parse_args()
config = fetch_config()["transformer"]
config.update(vars(args))
# --- fetch a pre-trained tokenizer from wandb -- #
tokenizer = fetch_tokenizer(config["tokenizer"])
# --- prepare the dataloaders --- #
train, val, _ = fetch_kor2eng(tokenizer.kor2eng)  # noqa
train = TensorDataset(
    P.to_src(tokenizer, config["max_length"], train),
    P.to_tgt_r(tokenizer, config["max_length"], train),
    P.to_tgt_ids(tokenizer, config["max_length"], train),
)
val = TensorDataset(
    P.to_src(tokenizer, config["max_length"], val),
    P.to_tgt_r(tokenizer, config["max_length"], val),
    P.to_tgt_ids(tokenizer, config["max_length"], val),
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
config.update({
    "vocab_size": tokenizer.get_vocab_size(),
    "pad_token_id": tokenizer.pad_token_id  # noqa
})
transformer = Transformer(**config)
# --- start wandb context --- #
with wandb.init(project="cleanformer", config=config, tags=[__file__]):
    # --- prepare a wandb_logger (wandb) and a trainer to use --- #
    wandb_logger = WandbLogger(log_model="all",
                               save_dir=WANDB_DIR)
    trainer = Trainer(
        fast_dev_run=config["fast_dev_run"],
        detect_anomaly=config['detect_anomaly'],
        limit_train_batches=config["limit_train_batches"],
        limit_val_batches=config["limit_val_batches"],
        check_val_every_n_epoch=config["check_val_every_n_epoch"],
        overfit_batches=config["overfit_batches"],
        log_every_n_steps=config["log_every_n_steps"],
        max_epochs=config["max_epochs"],
        gpus=torch.cuda.device_count(),
        logger=wandb_logger,
        callbacks=[
            ModelSummary(max_depth=config['max_depth']),
            ModelCheckpoint(
                verbose=config['verbose'],
                every_n_epochs=config["every_n_epochs"],
                save_on_train_epoch_end=config["save_on_train_epoch_end"],
            ),
            LearningRateMonitor(logging_interval="epoch"),
            LogCallback(tokenizer)
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

