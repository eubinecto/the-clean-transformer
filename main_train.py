import os
import torch
import wandb
import argparse
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from enkorde.models import Transformer
from enkorde.fetchers import fetch_tokenizer, fetch_config
from enkorde.paths import ROOT_DIR
from enkorde.datamodules import Kor2EngDataModule, Kor2EngSmallDataModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("entity", type=str)
    parser.add_argument("--ver", type=str, default="overfit_small_scratch")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    parser.add_argument("--overfit_batches", type=int, default=0)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=5)
    args = parser.parse_args()
    config = fetch_config()['train'][args.ver]
    config.update(vars(args))
    with wandb.init(entity=config['entity'], project="enkorde", config=config) as run:
        # --- fetch a pre-trained tokenizer from wandb -- #
        tokenizer = fetch_tokenizer(config['entity'], config['tokenizer'])
        # --- instantiate the transformer to train --- #
        transformer = Transformer(config['hidden_size'],
                                  config['ffn_size'],
                                  tokenizer.get_vocab_size(),  # vocab_size
                                  config['max_length'],
                                  tokenizer.pad_token_id,  # noqa
                                  config['heads'],
                                  config['depth'],
                                  config['dropout'],
                                  config['lr'])
        # --- choose the data --- #
        if config['data'] == Kor2EngDataModule.name:
            datamodule = Kor2EngDataModule(config, tokenizer)
        elif config['data'] == Kor2EngSmallDataModule.name:
            datamodule = Kor2EngSmallDataModule(config, tokenizer)
        else:
            raise ValueError(f"Invalid data: {config['data']}")
        # --- prepare a logger (wandb) and a trainer to use --- #
        logger = WandbLogger(log_model=False)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        trainer = Trainer(fast_dev_run=config['fast_dev_run'],
                          check_val_every_n_epoch=config['check_val_every_n_epoch'],
                          overfit_batches=config['overfit_batches'],
                          max_epochs=config['max_epochs'],
                          log_every_n_steps=config['log_every_n_steps'],
                          gpus=torch.cuda.device_count(),
                          callbacks=[lr_monitor],
                          enable_checkpointing=False,
                          logger=logger)
        # --- start training --- #
        trainer.fit(model=transformer, datamodule=datamodule)
        # save them only if the training is properly done
        if not config['fast_dev_run'] and trainer.current_epoch == config['max_epochs'] - 1:
            ckpt_path = os.path.join(ROOT_DIR, "transformer.ckpt")
            trainer.save_checkpoint(ckpt_path)
            artifact = wandb.Artifact(name="transformer", type="model", metadata=config)
            artifact.add_file(ckpt_path)
            run.log_artifact(artifact, aliases=["latest", config['ver']])
            os.remove(ckpt_path)  # make sure you remove it after you are done with uploading it


if __name__ == '__main__':
    main()
