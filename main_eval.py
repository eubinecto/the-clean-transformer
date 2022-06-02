"""
bleu 스코어 계산 필요.
"""
import argparse
import os
import torch  # noqa
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset  # noqa
from cleanformer.fetchers import fetch_kor2eng, fetch_config, fetch_tokenizer, fetch_transformer
from cleanformer import preprocess as P  # noqa

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    args = parser.parse_args()
    kor2eng = fetch_kor2eng()
    config = fetch_config()["transformer"]
    config.update(vars(args))
    tokenizer = fetch_tokenizer(config["tokenizer"])
    transformer = fetch_transformer(config["ver"], tokenizer)
    test = TensorDataset(
        P.src(tokenizer, config["max_length"], kor2eng[2]),
        P.tgt_r(tokenizer, config["max_length"], kor2eng[2]),
        P.tgt(tokenizer, config["max_length"], kor2eng[2]),
    )
    test_dataloader = DataLoader(
        test,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    # --- start wandb context --- #
    with wandb.init(project="cleanformer", config=config):
        # --- prepare a logger (wandb) and a trainer to use --- #
        logger = WandbLogger()
        trainer = Trainer(
            max_epochs=1,  # test over one epoch
            fast_dev_run=config["fast_dev_run"],
            gpus=None,  # do not use GPUs' for this
            logger=logger,
        )
        # start testing here
        trainer.test(model=transformer, dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
