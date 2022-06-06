"""
bleu 스코어 계산 필요.
"""
import argparse
import os
import shutil
import torch  # noqa
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset  # noqa
from cleanformer.logcallback import LogCallback
from cleanformer.fetchers import fetch_kor2eng, fetch_config, fetch_tokenizer, fetch_transformer
from cleanformer import preprocess as P  # noqa

os.environ["TOKENIZERS_PARALLELISM"] = "true"


parser = argparse.ArgumentParser()
required = parser.add_argument_group("required arguments")
required.add_argument("--batch_size", type=int, required=True)
optional = parser.add_argument_group("optional arguments")
optional.add_argument("--fast_dev_run", action="store_true", default=False)
optional.add_argument("--num_workers", type=int, default=os.cpu_count())
args = parser.parse_args()
config = fetch_config()["recommended"]
config.update(vars(args))
transformer = fetch_transformer(config["transformer"])
tokenizer = fetch_tokenizer(config["tokenizer"])
_, _, test = fetch_kor2eng(tokenizer.kor2eng)  # noqa
test = TensorDataset(
    P.to_src(tokenizer, transformer.hparams['max_length'], test),
    P.to_tgt_r(tokenizer, transformer.hparams['max_length'], test),
    P.to_tgt_ids(tokenizer, transformer.hparams['max_length'], test),
)
test_dataloader = DataLoader(
    test,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
)
# --- start wandb context --- #
with wandb.init(project="cleanformer", config=config, tags=[__file__]):
    # --- prepare a logger (wandb) and a trainer to use --- #
    logger = WandbLogger()
    trainer = Trainer(
        max_epochs=1,  # test over one epoch
        fast_dev_run=config["fast_dev_run"],
        gpus=torch.cuda.device_count(),
        logger=logger,
        callbacks=[LogCallback(tokenizer)],
    )
    # start testing here
    trainer.test(model=transformer, dataloaders=test_dataloader)
shutil.rmtree("wandb")


